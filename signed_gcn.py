import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, global_mean_pool


class signedGCN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()

        self.gcn_layer1 = GCNConv(in_channels, hid_channels)
        self.gcn_layer2 = GCNConv(hid_channels, out_channels)
        self.mlp = torch.nn.Linear(out_channels, 2)

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_weights, data.batch
        x = torch.nn.LeakyReLU(self.gcn_layer1(x, edge_index))
        x = torch.nn.LeakyReLU(self.gcn_layer2(x, edge_index))
        x = global_mean_pool(x, batch)  # Global Mean Pooling
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x



class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    # def forward(self, x, edge_index):
    #     # x has shape [N, in_channels]
    #     # edge_index has shape [2, E]

    #     # Step 1: Add self-loops to the adjacency matrix.
    #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    #     # Step 2: Linearly transform node feature matrix.
    #     x = self.lin(x)

    #     # Step 3: Compute normalization.
    #     row, col = edge_index
    #     deg = degree(col, x.size(0), dtype=x.dtype)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    #     # Step 4-5: Start propagating messages.
    #     out = self.propagate(edge_index, x=x, norm=norm)

    #     # Step 6: Apply a final bias vector.
    #     out = out + self.bias

    def forward(self, x, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # print(edge_weight)
        # print(edge_index)
        #edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        #print(edge_weight)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization.
        # weighted_adj =  to_dense_adj(edge_index, edge_weight)
        # print(weighted_adj)
        # print(edge_weight)
        deg = scatter(edge_weight.abs(), edge_index[0], reduce="sum")
        # deg = to_dense_adj(edge_index, edge_weight).abs().sum(dim=1)
        # print(deg)
        row, col = edge_index
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
