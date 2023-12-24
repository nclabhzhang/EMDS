import torch
import torch.nn.functional as F

from models.egnn import EGNN
from models.layers import MLP
from utils.graph_utils import mask_x, pow_tensor, remove_mean_with_mask
from models.attention import AttentionLayer


class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, n_layers_egnn, n_hid_egnn, norm_factor, aggregation_method):
        super(ScoreNetworkX, self).__init__()
        self.nfeat = max_feat_num
        self.pfeat = 3
        self.depth = depth
        self.nhid = nhid

        self._edge_dict = {}

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(EGNN(
                    in_node_nf=self.nfeat + 1, out_node_nf=self.nhid, in_edge_nf=1,       # node_dim + time_dim(1)
                    hidden_nf=n_hid_egnn, act_fn=torch.nn.SiLU(),
                    n_layers=n_layers_egnn, attention=True, tanh=True, norm_constant=1.0,
                    inv_sublayers=1, sin_embedding=False,
                    normalization_factor=norm_factor, aggregation_method=aggregation_method))
            else:
                self.layers.append(EGNN(
                    in_node_nf=self.nhid, out_node_nf=self.nhid, in_edge_nf=1,
                    hidden_nf=n_hid_egnn, act_fn=torch.nn.SiLU(),
                    n_layers=n_layers_egnn, attention=True, tanh=True, norm_constant=1.0,
                    inv_sublayers=1, sin_embedding=False,
                    normalization_factor=norm_factor, aggregation_method=aggregation_method))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, pos, adj, flags, t):
        batch_size, n_nodes, _ = x.shape
        edges = self.get_adj_matrix(batch_size, n_nodes, x.device)

        node_mask = flags.view(-1, 1)
        edge_mask = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes, n_nodes)).repeat(batch_size, 1, 1).view(-1, 1).to(x.device)

        x_time = t.view(batch_size, 1).repeat(1, n_nodes).unsqueeze(-1)
        x = torch.cat([x, x_time], dim=-1)

        x_list = [x[:, :, :-1]]
        for _ in range(self.depth):
            x, pos = x.view(batch_size * n_nodes, -1), pos.view(batch_size * n_nodes, -1)
            x, pos_ = self.layers[_](x, pos, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=adj.view(batch_size * n_nodes * n_nodes, -1))

            pos = (pos_ - pos) * node_mask  # This masking operation is redundant but just in case
            pos = pos.view(batch_size, n_nodes, -1)
            pos = remove_mean_with_mask(pos, flags.unsqueeze(-1))

            x = x.view(batch_size, n_nodes, -1)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1)   # B x N x (F + num_layers x H)
        x = self.final(xs).view(batch_size, n_nodes, -1)

        x = mask_x(x, flags)

        return x

    def get_adj_matrix(self, batch_size, n_nodes, device):
        if n_nodes in self._edge_dict:
            edges_dic_b = self._edge_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edge_dict[n_nodes] = {}
            return self.get_adj_matrix(batch_size, n_nodes, device)



class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears, c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()
        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x
