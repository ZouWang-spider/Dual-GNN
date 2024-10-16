import numpy as np

def get_sentiment_positions(true_label_indices, sentiment_labels_map, sentence_len):
    # 用于存储情感在表格中的坐标位置
    sentiment_positions = []

    # 遍历上三角区域的所有元素（不包括对角线）
    for i in range(sentence_len):
        for j in range(i + 1, sentence_len):  # 只遍历上三角区域
            # 如果当前位置是情感标签
            if true_label_indices[i, j] in sentiment_labels_map.values():
                # 找到情感在表格中的位置，存储在列表中
                sentiment_positions.append((i, j))

    return sentiment_positions

# 示例标签索引矩阵和情感标签映射
true_label_indices = np.array([
    [1, 0, 6, 0, 6],
    [0, 3, 6, 6, 6],
    [6, 6, 0, 6, 6],
    [0, 6, 6, 3, 6],
    [6, 6, 6, 6, 0]
])

sentiment_labels_map = {'POS': 0, 'NEU': 1, 'NEG': 2}
sentence_len = true_label_indices.shape[0]

# 获取情感位置
sentiment_positions = get_sentiment_positions(true_label_indices, sentiment_labels_map, sentence_len)
print("情感位置坐标:")
print(sentiment_positions)
