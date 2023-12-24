import torch
import torch.nn.functional as F

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_adjs, mask_x, pow_tensor, node_feature_to_matrix
from models.attention import AttentionLayer


class BaselineNetworkLayer(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):
        super(BaselineNetworkLayer, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))

        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2*conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:,_,:,:])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list , dim=-1)), flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads=4, conv='EGNN', n_layers_egnn=1, normalization_factor=4, aggregation_method='sum'):
        super(BaselineNetwork, self).__init__()
        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.n_layers_egnn = n_layers_egnn
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))
            elif _ == self.num_layers-1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))
            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)   

    def forward(self, x, adj, flags=None):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score



class ScoreNetworkA(BaselineNetwork):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads=4, conv='EGNN', n_layers_egnn=1, normalization_factor=4, aggregation_method='sum'):
        super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads=4, conv='EGNN', n_layers_egnn=1, normalization_factor=4, aggregation_method='sum')
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv

        self._edge_dict = {}

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat + 1, self.nhid, self.nhid,
                                                  self.c_init, self.c_hid,  self.num_heads,
                                                  self.conv, self.n_layers_egnn, self.normalization_factor, self.aggregation_method))
            elif _ < self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid,
                                                  self.c_hid, self.c_hid, self.num_heads,
                                                  self.conv, self.n_layers_egnn, self.normalization_factor,
                                                  self.aggregation_method))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid,
                                                  self.c_hid, self.c_final, self.num_heads,
                                                  self.conv, self.n_layers_egnn, self.normalization_factor,
                                                  self.aggregation_method))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  

    def forward(self, x, pos, adj, flags, t):
        batch_size, n_nodes, _ = x.shape
        edges = self.get_adj_matrix(batch_size, n_nodes, x.device)

        node_mask = flags.view(-1, 1)
        edge_mask = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes, n_nodes)).repeat(batch_size, 1, 1).view(-1,1).to(x.device)

        x_time = t.view(batch_size, 1).repeat(1, n_nodes).unsqueeze(-1)
        x = torch.cat([x, x_time], dim=-1)

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, pos, adjc = self.layers[_](x, pos, adjc, edges, flags, node_mask, edge_mask)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1]   # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


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