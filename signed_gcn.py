import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, global_mean_pool


class signedGCN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()

        self.gcn_layer1 = SignedGCNConv(in_channels, hid_channels)
        self.gcn_layer2 = SignedGCNConv(hid_channels, out_channels)
        self.mlp = torch.nn.Linear(out_channels, 2)
        self.sigma = torch.nn.LeakyReLU()

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = self.sigma(self.gcn_layer1(x, edge_index, edge_weights))
        x = self.sigma(self.gcn_layer2(x, edge_index, edge_weights))
        x = global_mean_pool(x, batch)  # Global Mean Pooling
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x



class SignedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight):
        x = self.lin(x)
        deg = scatter(edge_weight.abs(), edge_index[0], reduce="sum")
        row, col = edge_index
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()

        self.gcn_layer1 = GCNConv(in_channels, hid_channels)
        self.gcn_layer2 = GCNConv(hid_channels, out_channels)
        self.mlp = torch.nn.Linear(out_channels, 2)
        self.sigma = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_weight, data.batch
        edge_weights[edge_weights < 0] = 0
        x = self.sigma(self.gcn_layer1(x, edge_index, edge_weights))
        x = self.sigma(self.gcn_layer2(x, edge_index, edge_weights))
        x = global_mean_pool(x, batch)  # Global Mean Pooling
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x



