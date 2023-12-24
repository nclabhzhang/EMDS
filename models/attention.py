import math
import torch
import torch.nn.functional as F

from models.egnn import EGNN
from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_x, mask_pos, mask_adjs, remove_mean_with_mask


# -------- Graph Multi-Head Attention --------
class Attention(torch.nn.Module):
    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='EGNN', egnn_layers=1, normalization_factor=4, aggregation_method='sum'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        self.normalization_factor = normalization_factor

        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv, egnn_layers, normalization_factor, aggregation_method)
        self.activation = torch.tanh 
        self.softmax_dim = 2

    def forward(self, x, pos, adj, edges, flags, node_mask, edge_mask, attention_mask=None):
        batch_size, n_nodes, _ = x.shape
        x, pos = x.view(batch_size * n_nodes, -1), pos.view(batch_size * n_nodes, -1)
        if self.conv == 'EGNN':
            Q_x, Q_p = self.gnn_q(x, pos, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=adj.contiguous().view(batch_size*n_nodes*n_nodes, -1))
            K_x, K_p = self.gnn_k(x, pos, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=adj.contiguous().view(batch_size*n_nodes*n_nodes, -1))
        else:
            Q = self.gnn_q(x) 
            K = self.gnn_k(x)

        V_x, V_p = self.gnn_v(x, pos, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=adj.contiguous().view(batch_size*n_nodes*n_nodes, -1))

        Q_x, Q_p = Q_x.view(batch_size, n_nodes, -1), Q_p.view(batch_size, n_nodes, -1)
        K_x, K_p = K_x.view(batch_size, n_nodes, -1), K_p.view(batch_size, n_nodes, -1)
        V_x, V_p = V_x.view(batch_size, n_nodes, -1), V_p.view(batch_size, n_nodes, -1)

        Q_p = remove_mean_with_mask(Q_p, flags.unsqueeze(-1))
        K_p = remove_mean_with_mask(K_p, flags.unsqueeze(-1))
        V_p = remove_mean_with_mask(V_p, flags.unsqueeze(-1))

        Q = torch.cat([Q_x, Q_p], dim=-1)
        K = torch.cat([K_x, K_p], dim=-1)
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim) ) # (B x num_heads) x N x N
        
        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1,-2))/2 

        return V_x, V_p, A

    def get_gnn(self, in_dim, attn_dim, out_dim, conv='EGNN', egnn_layers=1, normalization_factor=4, aggregation_method='sum'):
        if conv == 'EGNN':
            gnn_q = EGNN(
                    in_node_nf=in_dim, out_node_nf=out_dim-3, in_edge_nf=1,
                    hidden_nf=attn_dim, act_fn=torch.nn.SiLU(),
                    n_layers=egnn_layers, attention=True, tanh=True, norm_constant=1.0,
                    inv_sublayers=1, sin_embedding=False,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method)
            gnn_k = EGNN(
                    in_node_nf=in_dim, out_node_nf=out_dim-3, in_edge_nf=1,
                    hidden_nf=attn_dim, act_fn=torch.nn.SiLU(),
                    n_layers=egnn_layers, attention=True, tanh=True, norm_constant=1.0,
                    inv_sublayers=1, sin_embedding=False,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method)
            gnn_v = EGNN(
                    in_node_nf=in_dim, out_node_nf=out_dim, in_edge_nf=1,
                    hidden_nf=attn_dim, act_fn=torch.nn.SiLU(),
                    n_layers=egnn_layers, attention=True, tanh=True, norm_constant=1.0,
                    inv_sublayers=1, sin_embedding=False,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method)
            return gnn_q, gnn_k, gnn_v

        elif conv == 'MLP':
            num_layers = 2
            gnn_q = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_k = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_v = DenseGCNConv(in_dim, out_dim)
            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f'{conv} not implemented.')


# -------- Layer of ScoreNetwork_A --------
class AttentionLayer(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim,
                 input_dim, output_dim, num_heads=4,
                 conv='EGNN', egnn_layers=1,
                 normalization_factor=4,
                 aggregation_method='sum'):
        super(AttentionLayer, self).__init__()
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(Attention(conv_input_dim, self.attn_dim, conv_output_dim,
                                       num_heads=num_heads,
                                       conv=conv,
                                       egnn_layers=egnn_layers,
                                       normalization_factor=normalization_factor,
                                       aggregation_method=aggregation_method
                                       )
                             )

        self.hidden_dim = 2 * max(input_dim, output_dim)
        self.mlp = MLP(num_linears, 2*input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel_x = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel_pos = MLP(2, input_dim*3, self.hidden_dim, 3, use_bn=False, activate_func=F.elu)

    def forward(self, x, pos, adj, edges, flags, node_mask, edge_mask):
        mask_list = []
        x_list = []
        pos_list = []

        for _ in range(len(self.attn)):
            _x, _pos, mask = self.attn[_](x, pos, adj[:,_,:,:], edges, flags, node_mask, edge_mask)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
            pos_list.append(_pos)

        x_out = mask_x(self.multi_channel_x(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)
        pos_out = mask_pos(self.multi_channel_pos(torch.cat(pos_list, dim=-1)), flags)
        pos_out = torch.tanh(pos_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, pos_out, adj_out
