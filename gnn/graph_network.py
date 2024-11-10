from typing import Any, Dict, List, Tuple
import torch
from torch_geometric.data import Data

from gnn.gcn import GCN

class GraphNetwork(object):
    def __init__(self, genome_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN(genome_config.num_inputs, genome_config.num_hidden, genome_config.num_outputs).to(self.device)


    def activate(self, nodes: List[Dict[str, Any]], edges: List[Tuple[int, int]]):
        edge_index = torch.tensor([[t[0] for t in edges], [t[1] for t in edges]], dtype=torch.long)
        x = torch.tensor([node.values() for node in nodes], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index).to(self.device)

        with torch.no_grad():
            out = self.model(data)
        return out.cpu().numpy()


    def set_parameters(self, genome_config, genome):
        conns_in_conv1 = genome_config.num_inputs*genome_config.num_hidden

        new_weight_values = torch.tensor([conn.weight for conn in list(genome.connections.values())[:conns_in_conv1]]).to(self.device)
        new_bias_values = torch.tensor([node.bias for node in list(genome.nodes.values())[genome_config.num_outputs:]]).to(self.device)
        self.model.conv1.lin.weight = torch.nn.Parameter(new_weight_values)
        self.model.conv1.bias = torch.nn.Parameter(new_bias_values)

        new_weight_values = torch.tensor([conn.weight for conn in list(genome.connections.values())[conns_in_conv1:]]).to(self.device)
        new_bias_values = torch.tensor([node.bias for node in list(genome.nodes.values())[:genome_config.num_outputs]]).to(self.device)
        self.model.conv2.lin.weight = torch.nn.Parameter(new_weight_values)
        self.model.conv2.bias = torch.nn.Parameter(new_bias_values)


    @staticmethod
    def create(genome, config):
        # TODO load activation function from config
        network = GraphNetwork(config.genome_config)
        network.set_parameters(config.genome_config, genome)
        return network


