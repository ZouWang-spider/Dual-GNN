import ast
import numpy as np

# 打开并读取train.txt文件
def Label_to_Table(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    sentences = []
    label_list = []
    # 处理每个元素
    for data in lines:
        parts = data.split('####')
        sentence = parts[0]  # 提取句子
        labels = parts[-1]  # 提取最后的标注部分
        sentences.append(sentence)
        label_list.append(labels)

    # 存储所有表格的列表
    all_tables = []
    for sentence, labels_str in zip(sentences, label_list):
        # 将句子拆分为单词列表
        words = sentence.split()
        n = len(words)  # 计算单词总数，包括标点符号

        # 创建n x n的二维表格，并初始化为空
        table = np.full((n, n), 'N', dtype=object)
        # 字符串类型转化为对象
        labels = ast.literal_eval(labels_str)

        # 提取具体的元素
        for entry in labels:
            aspect_indices, opinion_indices, sentiment = entry

            # 填充方面词（Aspect）
            if len(aspect_indices) == 1:
                table[aspect_indices[0]][aspect_indices[0]] = 'B-A'
            else:
                table[aspect_indices[0]][aspect_indices[0]] = 'B-A'
                for i in range(1, len(aspect_indices)):
                    table[aspect_indices[i]][aspect_indices[i]] = 'I-A'

            # 填充观点词（Opinion）
            if len(opinion_indices) == 1:
                table[opinion_indices[0]][opinion_indices[0]] = 'B-O'
            else:
                table[opinion_indices[0]][opinion_indices[0]] = 'B-O'
                for i in range(1, len(opinion_indices)):
                    table[opinion_indices[i]][opinion_indices[i]] = 'I-O'

            # 填充情感极性（Sentiment）
            for ai in aspect_indices:
                for oi in opinion_indices:
                    table[ai][oi] = sentiment
                    table[oi][ai] = sentiment  # 确保对称

        # 将每个表格加入列表
        all_tables.append(table)
    # 返回所有表格
    return all_tables


def Dataset_Sentence(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    sentences = []
    # 处理每个元素
    for data in lines:
        parts = data.split('####')
        sentence = parts[0]  # 提取句子
        sentences.append(sentence)

    return sentences

# path = "E:/PythonProject2/DualHGNN/triplet_data/16res/train.txt"
# table = Label_to_Table(path)
# print("label transform table_filling:")
# for row in table:
#     print(row)

# list = Dataset_Sentence(path)
# for sample in list:
#     print(sample)



