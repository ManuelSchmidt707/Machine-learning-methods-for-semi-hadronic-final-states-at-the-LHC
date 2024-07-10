import torch
import torch.nn as nn

import torch_geometric

from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F


class GCN_model_v2(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=238, output_dim = 2, num_layers = 2, dropout = 0.3):
        super(GCN_model_v2, self).__init__()
        layers = []
        layers.append((GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append((GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
        
        self.layers = torch_geometric.nn.Sequential('x, edge_index', layers)

        self.output_network = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layers(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.output_network(x)
        return x
    
class GATv2_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, heads = 1, output_dim = 2, num_layers = 2, negative_slope = 0.2, dropout = 0.3, output_hidden_dim = -1):
        super(GATv2_model, self).__init__()
        if output_hidden_dim == -1:
            output_hidden_dim = hidden_dim*heads
        layers = []
        layers.append((GATv2Conv(input_dim, hidden_dim, heads=heads, negative_slope =negative_slope, dropout= dropout), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append((GATv2Conv(heads * hidden_dim, hidden_dim, heads=heads, negative_slope =negative_slope, dropout= dropout), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
        
        self.output_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, output_hidden_dim),
            torch.nn.BatchNorm1d(output_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(output_hidden_dim, output_hidden_dim),
            torch.nn.BatchNorm1d(output_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(output_hidden_dim, output_dim)
            )

        self.layers = torch_geometric.nn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layers(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.output_network(x)
        return x
    
class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)

settings = {
    "conv_params": [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ],
    "fc_params": [
        (0.1, 256)
    ],
    "input_features": 10,
    "output_classes": 2,
}

class ParticleNet(torch.nn.Module):

    def __init__(self, settings):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K))
            previous_output_shape = channels[-1]



        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            )
            self.fc_process.append(seq)
            previous_output_shape = units


        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        for idx, layer in enumerate(self.conv_process):
            fts = layer(pts, fts, batch.batch)
            pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch.batch)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        x = self.output_activation(x)
        return x