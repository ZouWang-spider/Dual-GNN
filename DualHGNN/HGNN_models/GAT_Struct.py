import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GCNConv

edge_types = [
    ('struct', 'level', 'struct'),
    ('struct', 'level', 'word'),
    ('word', 'dis', 'distance_feature'),
    ('distance_feature', 'dis', 'word'),
    ('word', 'posit', 'position_feature'),
    ('position_feature', 'posit', 'word')
]

# 定义异构图神经网络模型
class Struct_HeteroSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Struct_HeteroSAGE, self).__init__()
        self.conv1 = HeteroConv(
            {('struct', 'level', 'struct'): GATConv(in_channels, hidden_channels,add_self_loops=False),
             ('struct', 'level', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('word', 'dis', 'distance_feature'): GATConv(in_channels, hidden_channels,  add_self_loops=False),
             ('distance_feature', 'dis', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('word', 'posit', 'position_feature'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             ('position_feature', 'posit', 'word'): GATConv(in_channels, hidden_channels, add_self_loops=False),
             },
            aggr='mean'
        )
        self.conv2 = HeteroConv(
            {('struct', 'level', 'struct'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('struct', 'level', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'dis', 'distance_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('distance_feature', 'dis', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'posit', 'position_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('position_feature', 'posit', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             },
            aggr='mean'
        )
        self.conv3 = HeteroConv(
            {('struct', 'level', 'struct'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('struct', 'level', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'dis', 'distance_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('distance_feature', 'dis', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('word', 'posit', 'position_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             ('position_feature', 'posit', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
             },
            aggr='mean'
        )
        # self.conv4 = HeteroConv(
        #     {('struct', 'level', 'struct'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      ('struct', 'level', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      ('word', 'dis', 'distance_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      ('distance_feature', 'dis', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      ('word', 'posit', 'position_feature'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      ('position_feature', 'posit', 'word'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
        #      },
        #     aggr='mean'
        # )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # Output feature dimension per word

    def forward(self, x_dict, edge_index_dict):
        # 执行第一层卷积，得到每种节点类型的嵌入表示
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 执行第二层卷积
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 执行第三层卷积
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # # 执行第四层卷积
        # x_dict = self.conv4(x_dict, edge_index_dict)
        # x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 将'word'节点类型的特征进行池化或拼接，作为最终的输出表示
        word_embeddings = x_dict['word']

        # 将'word'嵌入表示输入到全连接层进行处理
        word_embeddings = self.fc(word_embeddings)

        # 如果你希望将每个单词的嵌入表示作为输出，可以直接返回
        return word_embeddings


# 实例化模型
from stanfordcorenlp import StanfordCoreNLP
from DualHGNN.BaseModel.Emebdding_Biaffine import BERT_Embedding
from DualHGNN.BaseModel.Embedding_CoreNLP import parse_to_tuple, parse_tree_to_edges, compute_distance_matrix, compute_position_distance_matrix
from DualHGNN.BaseModel.Struct_embedding import get_level_feature_embedding, get_distance_embedding, get_position_embedding
from DualHGNN.BaseModel.Struct_to_Hetero import Struct_to_heterogeneous

nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
sentence ='The fried chicken was terrible'
constituent_tree = nlp.parse(sentence)
# 构成树类型转化，字符串转化为元祖
constituent_tree = parse_to_tuple(constituent_tree)
distance_matrix = compute_distance_matrix(constituent_tree)
position_matrix = compute_position_distance_matrix(sentence)

#构建结构异构图
g, x_dict, edge_index_dict = Struct_to_heterogeneous(sentence, constituent_tree, distance_matrix, position_matrix)

#HGNN initialization and training
in_channels = 768
hidden_channels = 512
out_channels =  512
model = Struct_HeteroSAGE(in_channels, hidden_channels, out_channels)
output = model(x_dict, edge_index_dict)
print(output.shape)