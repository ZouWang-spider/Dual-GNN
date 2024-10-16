import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
from DualHGNN.BaseModel.Emebdding_Biaffine import BERT_Embedding
from DualHGNN.BaseModel.Syntax_embedding import get_dep_feature_embedding, get_pos_feature_embedding, get_distance_embedding
from DualHGNN.BaseModel.Emebdding_Biaffine import BiAffine

#Word--Dep_feature
def extract_dep_edges(dep_features):
    word_to_dep_feature_edges = ([], [])  # ('word', 'dep', 'dep_feature')
    dep_feature_to_word_edges = ([], [])  # ('dep_feature', 'dep', 'word')

    # 依存特征映射到整数索引
    dep_feature_map = {}
    current_idx = 0

    for i, row in enumerate(dep_features):
        for j, feature in enumerate(row):
            if feature != 'self' and feature != '-':
                if feature not in dep_feature_map:
                    dep_feature_map[feature] = current_idx
                    current_idx += 1

                feature_idx = dep_feature_map[feature]

                # ('word', 'dep', 'dep_feature') -> word i to dep_feature (feature_idx)
                word_to_dep_feature_edges[0].append(i)
                word_to_dep_feature_edges[1].append(feature_idx)

                # ('dep_feature', 'dep', 'word') -> dep_feature (feature_idx) to word j
                dep_feature_to_word_edges[0].append(feature_idx)
                dep_feature_to_word_edges[1].append(j)

    return word_to_dep_feature_edges, dep_feature_to_word_edges, dep_feature_map


#Word---Pos_feature
def extract_pos_edges(pos_features):
    word_to_pos_feature_edges = ([], [])  # ('word', 'pos', 'pos_feature')
    pos_feature_to_word_edges = ([], [])  # ('pos_feature', 'pos', 'word')
    # 词性特征映射到整数索引
    pos_feature_map = {}
    current_idx = 0

    for i, row in enumerate(pos_features):
        for j, feature in enumerate(row):
            if feature != 'self':  # 排除 self
                if feature not in pos_feature_map:
                    pos_feature_map[feature] = current_idx
                    current_idx += 1

                feature_idx = pos_feature_map[feature]

                # ('word', 'pos', 'pos_feature') -> word i to pos_feature (feature_idx)
                word_to_pos_feature_edges[0].append(i)
                word_to_pos_feature_edges[1].append(feature_idx)

                # ('pos_feature', 'pos', 'word') -> pos_feature (feature_idx) to word j
                pos_feature_to_word_edges[0].append(feature_idx)
                pos_feature_to_word_edges[1].append(j)

    return word_to_pos_feature_edges, pos_feature_to_word_edges, pos_feature_map

#word---distance
def extract_distance_edges(distance_features):
    word_to_dis_feature_edges = ([], [])  # ('word', 'dis', 'dis_feature')
    dis_feature_to_word_edges = ([], [])  # ('dis_feature', 'dis', 'word')

    # 距离特征映射到整数索引
    dis_feature_map = {}
    current_idx = 0

    for i, row in enumerate(distance_features):
        for j, distance in enumerate(row):
            if i != j:  # 跳过自环（对角线上的0值）
                if distance not in dis_feature_map:
                    dis_feature_map[distance] = current_idx
                    current_idx += 1

                feature_idx = dis_feature_map[distance]

                # ('word', 'dis', 'dis_feature') -> word i to dis_feature (feature_idx)
                word_to_dis_feature_edges[0].append(i)
                word_to_dis_feature_edges[1].append(feature_idx)

                # ('dis_feature', 'dis', 'word') -> dis_feature (feature_idx) to word j
                dis_feature_to_word_edges[0].append(feature_idx)
                dis_feature_to_word_edges[1].append(j)

    return word_to_dis_feature_edges, dis_feature_to_word_edges, dis_feature_map



#Construct Heterogeneous Graph
def Syntactic_to_heterogeneous(sentence, dep_features, pos_features, distance_features):

    # Dep feature for Heterogeneous Graph
    word_to_dep, dep_to_word, dep_feature_map = extract_dep_edges(dep_features)
    # print(dep_to_word)
    # 将字典的键转换为列表
    dep_feature_labels = list(dep_feature_map.keys())

    # POS feature for Heterogeneous Graph
    word_to_pos, pos_to_word, pos_feature_map = extract_pos_edges(pos_features)
    # print(pos_feature_map)
    # 将字典的键转换为列表
    pos_feature_labels = list(pos_feature_map.keys())

    # Dep tree distance for Heterogeneous Graph
    word_to_dis, dis_to_word, dis_feature_map = extract_distance_edges(distance_features)
    # print(dis_feature_map)
    dis_feature_labels = list(dis_feature_map.keys())

    # 初始化异构图
    g = dgl.heterograph({
        ('word', 'dep', 'dep_feature'): word_to_dep,
        ('word', 'pos', 'pos_feature'): word_to_pos,
        ('word', 'dis', 'dis_feature'): word_to_dis,
        ('dep_feature', 'dep', 'word'): dep_to_word,
        ('pos_feature', 'pos', 'word'): pos_to_word,
        ('dis_feature', 'dis', 'word'): dis_to_word
    })

    # 这里我们为每个依存特征节点获取嵌入向量，并赋值给图中的 dep_feature 节点
    word_feature = BERT_Embedding(sentence)
    dep_feature_tensor = torch.stack([get_dep_feature_embedding(f) for f in dep_feature_labels])
    pos_feature_tensor = torch.stack([get_pos_feature_embedding(f) for f in pos_feature_labels])
    dis_feature_tensor = torch.stack([get_distance_embedding(dis_feature_map, f) for f in dis_feature_labels])

    #异构图的节点特征赋予
    g.nodes['word'].data['feature'] = word_feature
    g.nodes['dep_feature'].data['feature'] = dep_feature_tensor
    g.nodes['pos_feature'].data['feature'] = pos_feature_tensor
    g.nodes['dis_feature'].data['feature'] = dis_feature_tensor

    # # 为每种类型的边添加特征
    # hetero_graph.edges['follows'].data['weight'] = torch.tensor([0.5, 0.7])
    # hetero_graph.edges['plays'].data['weight'] = torch.tensor([1.0, 1.0])
    # hetero_graph.edges['develops'].data['weight'] = torch.tensor([0.9, 0.9])

    x_dict = {
        'word': g.nodes['word'].data['feature'],  # word 节点的特征
        'dep_feature': g.nodes['dep_feature'].data['feature'],  # dep_feature 节点的特征
        'pos_feature': g.nodes['pos_feature'].data['feature'],  # pos_feature 节点的特征
        'dis_feature': g.nodes['dis_feature'].data['feature']  # dis_feature 节点的特征
    }

    # 提取不同类型的边索引
    edge_index_dict = {
        ('word', 'dep', 'dep_feature'): torch.stack(g.edges(etype=('word', 'dep', 'dep_feature')), dim=0),
        ('word', 'pos', 'pos_feature'): torch.stack(g.edges(etype=('word', 'pos', 'pos_feature')), dim=0),
        ('word', 'dis', 'dis_feature'): torch.stack(g.edges(etype=('word', 'dis', 'dis_feature')), dim=0),
        ('dep_feature', 'dep', 'word'): torch.stack(g.edges(etype=('dep_feature', 'dep', 'word')), dim=0),
        ('pos_feature', 'pos', 'word'): torch.stack(g.edges(etype=('pos_feature', 'pos', 'word')), dim=0),
        ('dis_feature', 'dis', 'word'): torch.stack(g.edges(etype=('dis_feature', 'dis', 'word')), dim=0)
    }

    return g, x_dict, edge_index_dict


# data_path = "E:/PythonProject2/DualHGNN/triplet_data/14lap/test.txt"
# from DualHGNN.DataProcess.Label_to_Table import Dataset_Sentence
# Sentence_list = Dataset_Sentence(data_path)
#
# for sentence in Sentence_list:
#     print(sentence)
#     dep_features, pos_features, distance_features = BiAffine(sentence)
#     g, x_dict, edge_index_dict = Syntactic_to_heterogeneous(sentence, dep_features, pos_features, distance_features)

# # #function test
# sentence ='The chocolate cake is delicious'
# dep_features, pos_features, distance_features = BiAffine(sentence)
# g, x_dict, edge_index_dict = Syntactic_to_heterogeneous(sentence, dep_features, pos_features, distance_features)
# print(g)
