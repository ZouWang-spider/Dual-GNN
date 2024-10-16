from stanfordcorenlp import StanfordCoreNLP
import re

#constituent_tree的字符串类型转化为元祖类型
def parse_to_tuple(s):
    # 去掉多余的换行符和空格
    s = s.replace('\r', '').replace('\n', '').strip()

    # 递归解析括号树的函数
    if not s.startswith('('):
        return s
    s = s[1:-1].strip()  # 去掉外层括号
    space_idx = s.find(' ')
    label = s[:space_idx]  # 提取当前节点的标签
    rest = s[space_idx + 1:].strip()  # 剩下部分
    children = []
    while rest:
        if rest.startswith('('):
            # 找到匹配的右括号
            open_brackets = 0
            for i, char in enumerate(rest):
                if char == '(':
                    open_brackets += 1
                elif char == ')':
                    open_brackets -= 1
                if open_brackets == 0:
                    # 完整子树
                    children.append(parse_to_tuple(rest[:i + 1]))
                    rest = rest[i + 1:].strip()
                    break
        else:
            # 处理单词或词性
            if ' ' in rest:
                word, rest = rest.split(' ', 1)
                children.append(word.strip())
            else:
                children.append(rest.strip())
                rest = ''
    return (label, *children)

# 提取元组中的元素并转换为两个列表
def edges_to_tuples(edge_list):
    # 使用 zip 解压元组并创建两个列表
    u, v = zip(*edge_list)
    return list(u), list(v)

###########################构成树边关系(构成节点，构成节点)(构成节点，单词节点)###############################
def parse_tree_to_edges(parse_tree):
    """
    解析构成树，提取构成节点与构成节点之间的边关系，以及构成节点与单词节点之间的边关系。
    返回两种边关系字典：word_to_dep (构成节点与构成节点之间)，word_to_pos (构成节点与单词节点之间)，
    同时返回构成节点的映射。
    """
    struct_to_struct = []
    struct_to_word = []
    struct_mapping = {}  # 用于保存构成节点的映射

    def add_edges_from_subtree(subtree, parent=None, parent_type='struct'):
        # 解析子树并添加边到图
        if isinstance(subtree, str):
            # 叶子节点（单词），如果是单词，不需要添加边
            return
        label = subtree[0]
        if len(subtree) == 2 and isinstance(subtree[1], str):
            # 叶子节点（单词）
            word = subtree[1]
            word_id = len(nodes['word'])
            nodes['word'].append(word_id)
            node_labels['word'].append(word)
            if parent is not None:
                struct_to_word.append((parent, word_id))  # 构成节点与单词节点的关系
        else:
            # 内部节点（构成节点）
            if label not in struct_mapping:
                node_id = len(nodes['struct'])
                nodes['struct'].append(node_id)
                node_labels['struct'].append(label)
                struct_mapping[label] = node_id  # 将构成节点映射到唯一ID
            node_id = struct_mapping[label]  # 获取该构成节点的ID
            if parent is not None:
                struct_to_struct.append((parent, node_id))  # 构成节点与构成节点的关系
            for child in subtree[1:]:
                add_edges_from_subtree(child, node_id)
    # 初始化节点集合
    nodes = {'struct': [], 'word': []}
    node_labels = {'struct': [], 'word': []}
    # 解析构成树
    add_edges_from_subtree(parse_tree)

    struct_to_struct = edges_to_tuples(struct_to_struct)
    struct_to_word = edges_to_tuples(struct_to_word)
    return struct_to_struct, struct_to_word, struct_mapping


############################构成树节点距离矩阵##############################
def get_paths(tree, current_path=None, paths=None):
    if current_path is None:
        current_path = []
    if paths is None:
        paths = {}

    label = tree[0]  # 当前节点标签

    # 如果子节点是词性和单词对，记录路径
    if len(tree) == 2 and isinstance(tree[1], str):
        word = tree[1]
        paths[word] = current_path + [label]
    else:
        # 遍历子节点
        for subtree in tree[1:]:
            get_paths(subtree, current_path + [label], paths)

    return paths

# 计算两个单词路径的最近公共节点
def find_common_path_length(path1, path2):
    # 找到最近的公共父节点
    min_len = min(len(path1), len(path2))
    for i in range(min_len):
        if path1[i] != path2[i]:
            return i
    return min_len

# 计算单词之间的节点距离矩阵
def compute_distance_matrix(tree):
    paths = get_paths(tree)
    words = list(paths.keys())
    n = len(words)
    distance_matrix = [[0] * n for _ in range(n)]
    # 计算每对单词的距离
    for i in range(n):
        for j in range(i + 1, n):
            path1, path2 = paths[words[i]], paths[words[j]]
            common_length = find_common_path_length(path1, path2)
            # 计算节点之间的距离: 两者各自路径长度减去2倍的公共路径长度
            distance = (len(path1) - common_length) + (len(path2) - common_length)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


############################相对位子矩阵##############################
def compute_position_distance_matrix(sentence):
    # 将句子拆分为单词列表
    words = sentence.split()
    n = len(words)

    # 初始化位置距离矩阵
    position_matrix = [[0] * n for _ in range(n)]

    # 计算每对单词的相对位置距离
    for i in range(n):
        for j in range(n):
            position_matrix[i][j] = abs(i - j)

    return position_matrix


# nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
# sentence ='it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous .'
#
# constituent_tree = nlp.parse(sentence)
#
# # # 构成树类型转化，字符串转化为元祖
# constituent_tree = parse_to_tuple(constituent_tree)
# # 解析构成树并提取边关系
# struct_to_struct, struct_to_word, struct_mapping = parse_tree_to_edges(constituent_tree)
# # 计算距离矩阵
# distance_matrix = compute_distance_matrix(constituent_tree)


# # 计算相对位置距离矩阵
# position_matrix = compute_position_distance_matrix(sentence)
#
# # Constituency Tree Visualization
# import nltk
# from nltk.tree import Tree
# # 将构成树字符串转换为 NLTK 的 Tree 对象
# nltk_tree = Tree.fromstring(nlp.parse(sentence).strip())
# # 绘制树状结构
# nltk_tree.draw()