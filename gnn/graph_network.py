# import time
import logging
import numpy as np
import torch
from torch_geometric.data import Data

from gnn.gcn import GCN


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("GraphNetwork")


class GraphNetwork(object):
    def __init__(self, genome_config):
        self.device = torch.device('cpu') # cpu has been about 300ms faster than gpu
        self.model = GCN(genome_config.num_inputs, genome_config.num_hidden, genome_config.num_outputs).to(self.device)


    def activate(self, nodes: np.ndarray, edges: np.ndarray):
        # data_start = time.perf_counter_ns()
        data = Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.int16)).to(self.device)
        # LOGGER.info(f"tensor creation: {(time.perf_counter_ns()-data_start)/1000000} ms")
        # infer_start = time.perf_counter_ns()
        with torch.no_grad():
            out = self.model(data)
        # LOGGER.info(f"inference: {(time.perf_counter_ns()-infer_start)/1000000} ms")
        return out.cpu().numpy()[0].tolist()



    def ensure_length_n(self, arr, n):
        if len(arr) < n:
            arr = np.tile(arr, (n // len(arr)) + 1)[:n]
        elif len(arr) > n:
            arr = arr[:n]
        return arr

    def set_parameters(self, genome_config, genome):
        # sanity check to make sure NEAT does not generate less params than the net needs
        conv1_params_cnt = genome_config.num_hidden * genome_config.num_inputs
        all_params_cnt = conv1_params_cnt + genome_config.num_hidden * genome_config.num_outputs
        weights = self.ensure_length_n(np.array([conn.weight for conn in list(genome.connections.values())]), all_params_cnt)
        biases = self.ensure_length_n(np.array([node.bias for node in list(genome.nodes.values())]), all_params_cnt)
        conv1_params_cnt = genome_config.num_hidden * genome_config.num_inputs

        conv1_weights = torch.tensor(weights[:conv1_params_cnt].reshape((genome_config.num_hidden, genome_config.num_inputs)), dtype=torch.float).to(self.device)
        conv1_biases = torch.tensor(biases[:genome_config.num_hidden], dtype=torch.float).to(self.device)
        self.model.conv1.lin.weight = torch.nn.Parameter(conv1_weights)
        self.model.conv1.bias = torch.nn.Parameter(conv1_biases)

        conv2_weights = torch.tensor(weights[conv1_params_cnt:].reshape((genome_config.num_outputs, genome_config.num_hidden)), dtype=torch.float).to(self.device)
        conv2_biases = torch.tensor(biases[genome_config.num_hidden:genome_config.num_hidden+genome_config.num_outputs], dtype=torch.float).to(self.device)
        self.model.conv2.lin.weight = torch.nn.Parameter(conv2_weights)
        self.model.conv2.bias = torch.nn.Parameter(conv2_biases)


    @staticmethod
    def create(genome, config):
        # TODO load activation function from config
        network = GraphNetwork(config.genome_config)
        network.set_parameters(config.genome_config, genome)
        return network


