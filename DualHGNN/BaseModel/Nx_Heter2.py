import networkx as nx
import matplotlib.pyplot as plt


# 示例
sentence = 'The fried chicken was terrible'
distance_matrix = [[0, 2, 2, 4, 5], [2, 0, 2, 4, 5], [2, 2, 0, 4, 5], [4, 4, 4, 0, 3], [5, 5, 5, 3, 0]]
position_matrix = [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]

constituent_features = ['S', 'NP', 'VP', 'ADJP']
words = ['The', 'fried', 'chicken', 'was', 'terrible']
distance = ['d2', 'd3', 'd4', 'd5']
position = ['p1', 'p2', 'p3', 'p4']

edge1 = [('S','NP'),('S','VP'),('VP','ADJP')]
edge2 = [('NP','The'),('NP','fried'),('NP','chicken'),('VP','was'),('ADJP','terrible')]

edge3 =[('The','d2'),('d2', 'fried'),('d2', 'chicken'),('The','d4'),('d4','was'),('The','d5'),('d5','terrible'),
        ('fried','d2'),('d2','The'),('d2','chicken'),('fried','d4'),('fried','d5'),('chicken','d2'),('chicken','d4'),('chicken','d5'),
        ('was','d4'),('d4','The'),('d4','fried'),('d4','chicken'),('was','d3'),('d3','terrible'),('terrible','d5'),('d5','The'),('d5','fried'),('d5','chicken'),('terrible','d3'),('d3','was')]

edge4 =[('The','p1'),('p1','fried'),('The','p2'),('p2','chicken'),('The','p3'),('p3','was'),('The','p4'),('p4','terrible'),
        ('fried','p1'),('p1','The'),('p1','chicken'),('fried','p2'),('p2','was'),('fried','p3'),('was','p2'),
        ('chicken','p2'),('p2','The'),('chicken','p1'),('p1','was'),('chicken','p2'),('p2','terrible'),
        ('was','p3'),('p3','The'),('was','p2'),('p2','fried'),('was','p1'),('p1','terrible'),
        ('terrible','p4'),('p4','The'),('terrible','p3'),('p3','fried'),('terrible','p2'),('terrible','p1'),('p1','was')
        ]


# 初始化图
G = nx.Graph()
G.add_nodes_from(words, type='word')
G.add_nodes_from(constituent_features, type='constituency features')
G.add_nodes_from(distance, type='constituency tree distance')
G.add_nodes_from(position, type='relative position relation')

# 添加构成特征之间的边（edge1）
G.add_edges_from(edge1, label='const')
G.add_edges_from(edge2, label='const')
G.add_edges_from(edge3, label='posit')
G.add_edges_from(edge4, label='distan')


lay = nx.spring_layout(G)  # 布局
# 提取边标签
edge_labels = nx.get_edge_attributes(G, 'label')

# 创建节点标签，确保每个节点都有其正确的标签
node_labels = {node: node for node in G.nodes()}

# 为不同类型的节点分配颜色
node_colors = []
for node in G.nodes(data=True):
        node_type = node[1].get('type', 'word')  # 默认为 'word'
        if node_type == 'word':
            node_colors.append('lightgreen')
        elif node_type == 'constituency features':
            node_colors.append('lightblue')
        elif node_type == 'constituency tree distance':
            node_colors.append('coral')
        elif node_type == 'relative position relation':
            node_colors.append('lightpink')

# 绘制节点、边和标签
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, lay, node_size=500, node_color=node_colors)
nx.draw_networkx_edges(G, lay, edgelist=G.edges(), width=2, alpha=0.6)
nx.draw_networkx_labels(G, lay, labels=node_labels, font_size=10)
nx.draw_networkx_edge_labels(G, lay, edge_labels=edge_labels, font_size=8)
plt.title('Constituency Heterogeneous Graph')
plt.show()

