import logging
import numpy as np
import torch
from torch_geometric.data import Data

from gnn.gcn import GCN


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("GraphNetwork")


class GraphNetwork(object):
    def __init__(self, genome_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN(genome_config.num_inputs, genome_config.num_hidden, genome_config.num_outputs).to(self.device)


    def activate(self, nodes: np.ndarray, edges: np.ndarray):
        data = Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.int8)).to(self.device)
        with torch.no_grad():
            out = self.model(data)
        return out.cpu().numpy()[0].tolist()


    def set_parameters(self, genome_config, genome):
        new_weight_values = torch.tensor(np.array([conn.weight for conn in list(genome.connections.values())]).reshape((genome_config.num_outputs, genome_config.num_inputs)), dtype=torch.float).to(self.device)
        new_bias_values = torch.tensor([node.bias for node in list(genome.nodes.values())], dtype=torch.float).to(self.device)
        self.model.conv1.lin.weight = torch.nn.Parameter(new_weight_values)
        self.model.conv1.bias = torch.nn.Parameter(new_bias_values)

        # new_weight_values = torch.tensor(np.array([conn.weight for conn in list(genome.connections.values())[conns_in_conv1:]]).reshape((genome_config.num_outputs, genome_config.num_hidden))).to(self.device)
        # new_bias_values = torch.tensor([node.bias for node in list(genome.nodes.values())[:genome_config.num_outputs]]).to(self.device)
        # self.model.conv2.lin.weight = torch.nn.Parameter(new_weight_values)
        # self.model.conv2.bias = torch.nn.Parameter(new_bias_values)


    @staticmethod
    def create(genome, config):
        # TODO load activation function from config
        network = GraphNetwork(config.genome_config)
        network.set_parameters(config.genome_config, genome)
        return network


