import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F


# 定义异构图神经网络模型
class HeteroSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HeteroSAGE, self).__init__()
        self.conv1 = HeteroConv(
            {('word', 'depends_on', 'word'): SAGEConv(in_channels, hidden_channels),
             ('word', 'sequence', 'sequence'): SAGEConv(in_channels, hidden_channels),
             ('word', 'has_pos', 'pos'): SAGEConv(in_channels, hidden_channels)},
            aggr='mean'
        )
        self.conv2 = HeteroConv(
            {('word', 'depends_on', 'word'): SAGEConv(hidden_channels, hidden_channels),
             ('word', 'sequence', 'sequence'): SAGEConv(hidden_channels, hidden_channels),
             ('word', 'has_pos', 'pos'): SAGEConv(hidden_channels, hidden_channels)},
            aggr='mean'
        )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # Output feature dimension per word

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Apply linear transformation to each word's features
        x_word = x_dict['word']  # Shape: [num_words, hidden_channels]

        # Apply the linear layer to each word's feature
        x_word = self.fc(x_word)  # Shape: [num_words, out_channels]
        print(x_word.shape)

        return x_word
