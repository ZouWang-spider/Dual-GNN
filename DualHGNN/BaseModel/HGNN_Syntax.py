import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

edge_types = [
    ('word', 'dep', 'dep_feature'),
    ('word', 'pos', 'pos_feature'),
    ('word', 'dis', 'dis_feature'),
    ('dep_feature', 'dep', 'word'),
    ('pos_feature', 'pos', 'word'),
    ('dis_feature', 'dis', 'word')
]

# 定义异构图神经网络模型
class Syntax_HeteroSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Syntax_HeteroSAGE, self).__init__()
        self.conv1 = HeteroConv(
            {('word', 'dep', 'dep_feature'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('word', 'pos', 'pos_feature'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('word', 'dis', 'dis_feature'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('dep_feature', 'dep', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('pos_feature', 'pos', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('dis_feature', 'dis', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             },
            aggr='mean'
        )
        self.conv2 = HeteroConv(
            {('word', 'dep', 'dep_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'pos', 'pos_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'dis', 'dis_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('dep_feature', 'dep', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('pos_feature', 'pos', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('dis_feature', 'dis', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             },
            aggr='mean'
        )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # Output feature dimension per word

    def forward(self, x_dict, edge_index_dict):
        # 执行第一层卷积，得到每种节点类型的嵌入表示
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 执行第二层卷积
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 将'word'节点类型的特征进行池化或拼接，作为最终的输出表示
        word_embeddings = x_dict['word']

        # 将'word'嵌入表示输入到全连接层进行处理
        word_embeddings = self.fc(word_embeddings)

        # 如果你希望将每个单词的嵌入表示作为输出，可以直接返回
        return word_embeddings


# # 实例化模型
# from DualHGNN.BaseModel.Emebdding_Biaffine import BiAffine
# from DualHGNN.BaseModel.Syntax_to_Hetero import Syntactic_to_heterogeneous
# sentence ='The chocolate cake is delicious'
# dep_features, pos_features, distance_features = BiAffine(sentence)
# g, x_dict, edge_index_dict = Syntactic_to_heterogeneous(sentence, dep_features, pos_features, distance_features)
#
# #HGNN initialization and training
# in_channels = 768
# hidden_channels = 512
# out_channels =  512
# model = Syntax_HeteroSAGE(in_channels, hidden_channels, out_channels)
# output = model(x_dict, edge_index_dict)
# print(output.shape)