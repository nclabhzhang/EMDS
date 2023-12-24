import torch
import torch.nn.functional as F

from models.egnn import EGNN
from models.layers import MLP
from utils.graph_utils import mask_pos, pow_tensor, remove_mean_with_mask
from models.attention import AttentionLayer


class ScoreNetworkP(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, n_layers_egnn, n_hid_egnn, norm_factor, aggregation_method):
        super(ScoreNetworkP, self).__init__()
        self.nfeat = max_feat_num
        self.pfeat = 3
        self.depth = depth
        self.nhid = nhid

        self._edge_dict = {}

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(EGNN(
                    in_node_nf=self.nfeat + 1, out_node_nf=self.nhid, in_edge_nf=1,
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

        self.final_pos_dim = self.pfeat + self.depth * self.pfeat
        self.final = MLP(num_layers=3, input_dim=self.final_pos_dim, hidden_dim=2*self.final_pos_dim, output_dim=self.pfeat, use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh


    def forward(self, x, pos, adj, flags, t):
        batch_size, n_nodes, _ = x.shape
        edges = self.get_adj_matrix(batch_size, n_nodes, x.device)

        node_mask = flags.view(-1, 1)
        edge_mask = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes, n_nodes)).repeat(batch_size, 1, 1).view(-1, 1).to(x.device)

        x_time = t.view(batch_size, 1).repeat(1, n_nodes).unsqueeze(-1)
        x = torch.cat([x, x_time], dim=-1)

        pos_list = [pos]
        for _ in range(self.depth):
            x, pos = x.view(batch_size * n_nodes, -1), pos.view(batch_size * n_nodes, -1)
            x, pos_ = self.layers[_](x, pos, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=adj.view(batch_size * n_nodes * n_nodes, -1))

            pos = (pos_ - pos) * node_mask  # This masking operation is redundant but just in case
            pos = pos.view(batch_size, n_nodes, -1)
            pos = remove_mean_with_mask(pos, flags.unsqueeze(-1))

            pos = self.activation(pos)
            pos_list.append(pos)

        poss = torch.cat(pos_list, dim=-1)  # B x N x (3 + num_layers x 3)
        pos = self.final(poss).view(batch_size, n_nodes, -1)

        pos = mask_pos(pos, flags)

        return pos


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




class ScoreNetworkP_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears, c_init, c_hid, c_final, adim, num_heads=4, conv='EGNN', n_layers_egnn=1, normalization_factor=4, aggregation_method='sum'):
        super().__init__()
        self.pfeat = 3
        self.depth = depth
        self.c_init = c_init

        self._edge_dict = {}

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid,
                                                  c_init, c_hid, num_heads,
                                                  conv, n_layers_egnn, normalization_factor, aggregation_method))
            elif _ < self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid,
                                                  c_hid, c_hid, num_heads,
                                                  conv, n_layers_egnn, normalization_factor, aggregation_method))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid,
                                                  c_hid, c_final, num_heads,
                                                  conv, n_layers_egnn, normalization_factor, aggregation_method))

        self.final_pos_dim = self.pfeat + self.depth * self.pfeat
        self.final = MLP(num_layers=3, input_dim=self.final_pos_dim, hidden_dim=2 * self.final_pos_dim, output_dim=self.pfeat, use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, pos, adj, flags):
        batch_size, n_nodes, _ = x.shape
        edges = self.get_adj_matrix(batch_size, n_nodes, x.device)

        adjc = pow_tensor(adj, self.c_init)

        pos_list = [pos]
        for _ in range(self.depth):
            x, pos, adjc = self.layers[_](x, pos, adjc, edges, flags)
            pos = self.activation(pos)
            pos_list.append(pos)

        poss = torch.cat(pos_list, dim=-1)   # B x N x (F + num_layers x H)
        out_shape = (batch_size, n_nodes, -1)
        pos = self.final(poss).view(*out_shape)
        pos = mask_pos(pos, flags)

        return pos

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
