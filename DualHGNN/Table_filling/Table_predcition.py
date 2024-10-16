import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTagPredictionModel(nn.Module):
    def __init__(self, hidden_dim, sentence_len):
        super(CrossTagPredictionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.sentence_len = sentence_len

        # 5 分类的全连接层，用于对角线标签预测
        self.fc_diag = nn.Linear(hidden_dim, 5)

        # 情感三分类的全连接层，用于非对角线位置的情感预测
        self.fc_sentiment = nn.Linear(hidden_dim, 3)

        # 标签映射：5分类标签
        self.diag_labels_map = {
            0: 'B-A',
            1: 'I-A',
            2: 'B-O',
            3: 'I-O',
            4: 'N',
        }

        # 情感标签映射：3分类标签
        self.sentiment_labels_map = {
            0: 'POS',
            1: 'NEU',
            2: 'NEG',
        }

    def forward(self, sentence_hidden):
        sentence_len = sentence_hidden.size(0)

        sentence_hidden_expanded_i = sentence_hidden.unsqueeze(1)  # (5, 1, 512)
        sentence_hidden_expanded_j = sentence_hidden.unsqueeze(0)  # (1, 5, 512)

        # 通过广播机制计算 (5, 5, 512) 形状的相关矩阵
        correlation_matrix = sentence_hidden_expanded_i * sentence_hidden_expanded_j
        # print(correlation_matrix.shape)

        # 初始化二维表格，形状为 (sentence_len, sentence_len)，全部初始化为 "N"
        predicted_labels = [["N" for _ in range(sentence_len)] for _ in range(sentence_len)]

        # 遍历对角线，预测对角线上的标签
        for i in range(sentence_len):
            # 对角线上通过全连接层预测5分类
            diag_logits = self.fc_diag(correlation_matrix[i][i])
            diag_pred_idx = torch.argmax(F.softmax(diag_logits, dim=-1)).item()
            diag_pred_label = self.diag_labels_map[diag_pred_idx]
            predicted_labels[i][i] = diag_pred_label

        # 检查多词方面词和观点词
        i = 0
        while i < sentence_len - 1:
            # 检查是否存在 B-A 后面跟多个 I-A
            if predicted_labels[i][i] == 'B-A':
                for j in range(i + 1, sentence_len):
                    if predicted_labels[j][j] == 'I-A':
                        predicted_labels[i][j] = 'A'  # 设置 (i, j) 为 'A'
                    else:
                        break  # 如果不再是 'I-A'，停止寻找
            # 检查是否存在 B-O 后面跟多个 I-O
            if predicted_labels[i][i] == 'B-O':
                for j in range(i + 1, sentence_len):
                    if predicted_labels[j][j] == 'I-O':
                        predicted_labels[i][j] = 'O'  # 设置 (i, j) 为 'O'
                    else:
                        break  # 如果不再是 'I-O'，停止寻找
            i += 1  # 移动到下一个对角线元素

        # 遍历对角线，寻找方面词和观点词的组合进行情感三分类预测
        for i in range(sentence_len):
            # 如果当前对角线是方面词标签 B-A 或 I-A
            if predicted_labels[i][i] in ['B-A', 'I-A']:
                # 遍历其他对角线，寻找观点词标签 B-O 或 I-O
                for j in range(i + 1, sentence_len):  # 只遍历上三角区域
                    if predicted_labels[j][j] in ['B-O', 'I-O']:
                        # 对 (i, j) 位置进行情感三分类预测
                        sentiment_logits = self.fc_sentiment(correlation_matrix[i, j])  # (512,)
                        sentiment_pred_idx = torch.argmax(F.softmax(sentiment_logits, dim=-1)).item()
                        sentiment_pred_label = self.sentiment_labels_map[sentiment_pred_idx]
                        # 在 (i, j) 位置赋予情感标签
                        predicted_labels[i][j] = sentiment_pred_label

        # 实现表格的斜对称
        for i in range(sentence_len):
            for j in range(i + 1, sentence_len):
                predicted_labels[j][i] = predicted_labels[i][j]

        return predicted_labels, correlation_matrix


# # 示例
# sentence_hidden = torch.randn(5, 512)  # 假设句子有5个单词，每个单词的隐藏向量维度为512
#
# model = CrossTagPredictionModel(hidden_dim=sentence_hidden.size(1), sentence_len=sentence_hidden.size(0))
# predicted_labels = model(sentence_hidden)
#
# # 打印预测的二维标签矩阵
# for row in predicted_labels:
#     print(row)
