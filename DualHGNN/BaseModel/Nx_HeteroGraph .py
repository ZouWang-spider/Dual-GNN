import dgl
import networkx as nx
import matplotlib.pyplot as plt
from DualHGNN.BaseModel.Emebdding_Biaffine import BiAffine


def Syntactic_to_heterogeneous(words, dep_features, pos_features, distance_features):
    # nodes = ([2, 2, 4, 4, 4], [0, 1, 2, 3, 4])
    # 创建一个有向图
    G = nx.DiGraph()
    # 添加原始依存图的节点
    G.add_nodes_from(range(len(words)), type='word')

    # Dep feature for Heterogeneous Graph
    # dep_features = [
    #     ['self', '-', 'det', '-', '-'],
    #     ['-', 'self', 'nn', '-', '-'],
    #     ['det', 'nn', 'self', '-', 'nsubj'],
    #     ['-', '-', '-', 'self', 'cop'],
    #     ['-', '-', 'nsubj', 'cop', 'self']
    # ]

    # 添加依存特征节点及其边    ('word', 'dep', 'dep_feature')  # word -> dep_feature
    for i, row in enumerate(dep_features):
        for j, feature in enumerate(row):
            if feature != 'self' and feature != '-':
                # 创建特征节点
                feature_node = f'{feature}'
                G.add_node(feature_node, type='dep_feature')
                # 连接原始节点和特征节点
                G.add_edge(i, feature_node, label='dep')
                G.add_edge(feature_node, j, label='dep')

    # POS feature for Heterogeneous Graph
    # pos_features = [
    #     ['self', 'DT-NN', 'DT-NN', 'DT-VBZ', 'DT-JJ'],
    #     ['NN-DT', 'self', 'NN-NN', 'NN-VBZ', 'NN-JJ'],
    #     ['NN-DT', 'NN-NN', 'self', 'NN-VBZ', 'NN-JJ'],
    #     ['VBZ-DT', 'VBZ-NN', 'VBZ-NN', 'self', 'VBZ-JJ'],
    #     ['JJ-DT', 'JJ-NN', 'JJ-NN', 'JJ-VBZ', 'self']
    # ]

    # 添加POS特征节点及其边   ('word', 'pos', 'pos_feature')  # word -> pos_feature 边
    for i, row in enumerate(pos_features):
        for j, feature in enumerate(row):
            if feature != 'self':
                pos_node = f'{feature}'
                G.add_node(pos_node, type='pos_feature')
                G.add_edge(i, pos_node, label='pos')
                G.add_edge(pos_node, j, label='pos')

    # Dep tree distance for Heterogeneous Graph
    # distance_features = [
    #     [0, 2, 1, 3, 2],
    #     [2, 0, 1, 3, 2],
    #     [1, 1, 0, 2, 1],
    #     [3, 3, 2, 0, 1],
    #     [2, 2, 1, 1, 0]
    # ]
    # 添加距离特征节点及其边     ('word', 'dis', 'dis_feature')   # word -> dis_feature 边
    for i, row in enumerate(distance_features):
        for j, distance in enumerate(row):
            if i != j:  # 自己到自己没有距离特征边
                dis_node = f'{distance}'
                G.add_node(dis_node, type='dis_feature')
                G.add_edge(i, dis_node, label='dis')
                G.add_edge(dis_node, j, label='dis')

    # Heterogeneous Graph  Visual
    lay = nx.spring_layout(G)  # 布局
    # edge_labels = {(start, end): '' for start, end in G.edges()}  # 边标签为空
    edge_labels = nx.get_edge_attributes(G, 'label')  # 获取边标签
    # node_labels = {node: node for node in G.nodes}
    plt.figure(figsize=(10, 6))
    # 更新节点标签为句子中的单词
    node_labels = {i: words[i] for i in range(len(words))}
    node_labels.update({node: node for node in G.nodes if node not in node_labels})
    # 为不同类型的节点分配颜色
    node_colors = []
    for node in G.nodes(data=True):
        node_type = node[1].get('type', 'word')  # 如果type不存在，默认为'word'
        if node_type == 'word':
            node_colors.append('lightgreen')
        elif node_type == 'dep_feature':
            node_colors.append('orange')
        elif node_type == 'pos_feature':
            node_colors.append('orchid')
        elif node_type == 'dis_feature':
            node_colors.append('coral')
    # 绘制节点
    nx.draw_networkx_nodes(G, lay, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, lay, edgelist=G.edges(), width=2, alpha=0.6)
    nx.draw_networkx_labels(G, lay, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G, lay, edge_labels=edge_labels)
    plt.title('Dependency Heterogeneous Graph')
    plt.show()
    return G





#function test
sentence ='The fried chicken was terrible'
dep_features, pos_features, distance_features = BiAffine(sentence)
words = sentence.split()
G = Syntactic_to_heterogeneous(words, dep_features, pos_features, distance_features)

