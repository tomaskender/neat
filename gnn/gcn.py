import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("GCN")


class GCN(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.conv1 = GCNConv(inputs, outputs)
        # self.conv2 = GCNConv(hidden, outputs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
        y = F.softmax(x, dim=1)
        return y