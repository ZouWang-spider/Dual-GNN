import dgl
import torch
from stanfordcorenlp import StanfordCoreNLP
from DualHGNN.BaseModel.Emebdding_Biaffine import BERT_Embedding
from DualHGNN.BaseModel.Embedding_CoreNLP import parse_to_tuple, parse_tree_to_edges, compute_distance_matrix, compute_position_distance_matrix
from DualHGNN.BaseModel.Struct_embedding import get_level_feature_embedding, get_distance_embedding, get_position_embedding

def  extract_distance_edges(distance_matrix):
    word_to_disfeature_edges = ([], [])
    disfeature_to_word_edges = ([], [])
    # 距离特征映射到整数索引
    disfeature_map = {}
    current_idx = 0
    for i, row in enumerate(distance_matrix):
        for j, distance in enumerate(row):
            if i != j:  # 跳过自环（对角线上的0值）
                if distance not in disfeature_map:
                    disfeature_map[distance] = current_idx
                    current_idx += 1

                feature_idx = disfeature_map[distance]

                # ('word', 'dis', 'dis_feature') -> word i to dis_feature (feature_idx)
                word_to_disfeature_edges[0].append(i)
                word_to_disfeature_edges[1].append(feature_idx)

                # ('dis_feature', 'dis', 'word') -> dis_feature (feature_idx) to word j
                disfeature_to_word_edges[0].append(feature_idx)
                disfeature_to_word_edges[1].append(j)
    return word_to_disfeature_edges, disfeature_to_word_edges, disfeature_map



# Realtive Position distance to edge
def extract_position_edges(distance_features):
    word_to_disfeature_edges = ([], [])
    disfeature_to_word_edges = ([], [])
    # 距离特征映射到整数索引
    disfeature_map = {}
    current_idx = 0
    for i, row in enumerate(distance_features):
        for j, distance in enumerate(row):
            if i != j:  # 跳过自环（对角线上的0值）
                if distance not in disfeature_map:
                    disfeature_map[distance] = current_idx
                    current_idx += 1

                feature_idx = disfeature_map[distance]

                # ('word', 'dis', 'dis_feature') -> word i to dis_feature (feature_idx)
                word_to_disfeature_edges[0].append(i)
                word_to_disfeature_edges[1].append(feature_idx)

                # ('dis_feature', 'dis', 'word') -> dis_feature (feature_idx) to word j
                disfeature_to_word_edges[0].append(feature_idx)
                disfeature_to_word_edges[1].append(j)
    return word_to_disfeature_edges, disfeature_to_word_edges, disfeature_map



#Construct Heterogeneous Graph
def Struct_to_heterogeneous(sentence, constituent_tree, distance_matrix, position_matrix):
    # 解析构成树并提取边关系
    struct_to_struct, struct_to_word, struct_mapping = parse_tree_to_edges(constituent_tree)
    struct_feature_labels = list(struct_mapping.keys())

    # # Contituent tree distance for Heterogeneous Graph
    word_to_distance, distance_to_word, distance_mapping = extract_distance_edges(distance_matrix)
    distance_feature_labels = list(distance_mapping.keys())

    # Relative Position feature for Heterogeneous Graph
    word_to_position, position_to_word, position_mapping = extract_position_edges(position_matrix)
    position_feature_labels = list(position_mapping.keys())

    # 初始化异构图
    g = dgl.heterograph({
        ('struct', 'level', 'struct'): struct_to_struct,
        ('struct', 'level', 'word'): struct_to_word,
        ('word', 'dis', 'distance_feature'): word_to_distance,
        ('distance_feature', 'dis', 'word'): distance_to_word,
        ('word', 'posit', 'position_feature'): word_to_position,
        ('position_feature', 'posit', 'word'): position_to_word
    })

    # print(len(g.nodes('word')))  # 检查图中 'word' 类型节点的数量

    # 这里我们为每个依存特征节点获取嵌入向量，并赋值给图中的 dep_feature 节点
    word_feature = BERT_Embedding(sentence)
    # print(word_feature.shape)
    struct_feature_tensor = torch.stack([get_level_feature_embedding(f) for f in struct_feature_labels])
    distance_feature_tensor = torch.stack([get_distance_embedding(distance_mapping, f) for f in distance_feature_labels])
    position_feature_tensor = torch.stack([get_position_embedding(position_mapping, f) for f in position_feature_labels])

    #异构图的节点特征赋予
    g.nodes['struct'].data['feature'] = struct_feature_tensor
    g.nodes['word'].data['feature'] = word_feature
    g.nodes['distance_feature'].data['feature'] = distance_feature_tensor
    g.nodes['position_feature'].data['feature'] = position_feature_tensor

    # # 为每种类型的边添加特征
    # hetero_graph.edges['follows'].data['weight'] = torch.tensor([0.5, 0.7])
    # hetero_graph.edges['plays'].data['weight'] = torch.tensor([1.0, 1.0])
    # hetero_graph.edges['develops'].data['weight'] = torch.tensor([0.9, 0.9])

    x_dict = {
        'word': g.nodes['word'].data['feature'],  # word 节点的特征
        'struct': g.nodes['struct'].data['feature'],  # dep_feature 节点的特征
        'distance_feature': g.nodes['distance_feature'].data['feature'],  # pos_feature 节点的特征
        'position_feature': g.nodes['position_feature'].data['feature']  # dis_feature 节点的特征
    }

    # 提取不同类型的边索引
    edge_index_dict = {
        ('struct', 'level', 'struct'): torch.stack(g.edges(etype=('struct', 'level', 'struct')), dim=0),
        ('struct', 'level', 'word'): torch.stack(g.edges(etype=('struct', 'level', 'word')), dim=0),
        ('word', 'dis', 'distance_feature'): torch.stack(g.edges(etype=('word', 'dis', 'distance_feature')), dim=0),
        ('distance_feature', 'dis', 'word'): torch.stack(g.edges(etype=('distance_feature', 'dis', 'word')), dim=0),
        ('word', 'posit', 'position_feature'): torch.stack(g.edges(etype=('word', 'posit', 'position_feature')), dim=0),
        ('position_feature', 'posit', 'word'): torch.stack(g.edges(etype=('position_feature', 'posit', 'word')), dim=0)
    }
    return g, x_dict, edge_index_dict


# nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
# data_path = "E:/PythonProject2/DualHGNN/triplet_data/14lap/test.txt"
# sentence = 'The fried chicken was terrible'
# constituent_tree = nlp.parse(sentence)
# # 构成树类型转化，字符串转化为元祖
# constituent_tree = parse_to_tuple(constituent_tree)
# print(constituent_tree)
# distance_matrix = compute_distance_matrix(constituent_tree)
# print(distance_matrix)
# position_matrix = compute_position_distance_matrix(sentence)
# print(position_matrix)

# # 构建结构异构图
# g, x_dict, edge_index_dict = Struct_to_heterogeneous(sentence, constituent_tree, distance_matrix, position_matrix)

# from DualHGNN.DataProcess.Label_to_Table import Dataset_Sentence
# Sentence_list = Dataset_Sentence(data_path)


# for sentence in Sentence_list:
#     # sentence = 'We , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .'
#     print("sentence:", sentence)
#     constituent_tree = nlp.parse(sentence)
#     # 构成树类型转化，字符串转化为元祖
#     constituent_tree = parse_to_tuple(constituent_tree)
#     distance_matrix = compute_distance_matrix(constituent_tree)
#     position_matrix = compute_position_distance_matrix(sentence)
#     # 构建结构异构图
#     g, x_dict, edge_index_dict = Struct_to_heterogeneous(sentence, constituent_tree, distance_matrix, position_matrix)

