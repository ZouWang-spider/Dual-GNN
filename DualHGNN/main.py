import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss
import torch.optim as optim

from DualHGNN.DataProcess.Label_to_Table import Dataset_Sentence


from DualHGNN.BaseModel.Emebdding_Biaffine import BiAffine
from DualHGNN.BaseModel.HGNN_Syntax import Syntax_HeteroSAGE
from DualHGNN.BaseModel.Syntax_to_Hetero import Syntactic_to_heterogeneous

from stanfordcorenlp import StanfordCoreNLP
from DualHGNN.BaseModel.Embedding_CoreNLP import parse_to_tuple, parse_tree_to_edges, compute_distance_matrix, compute_position_distance_matrix
from DualHGNN.BaseModel.Struct_to_Hetero import Struct_to_heterogeneous
from DualHGNN.BaseModel.HGNN_Struct import Struct_HeteroSAGE

from DualHGNN.Linguistic_Learning.Cross_Attention import CrossAttentionModule

from DualHGNN.Table_filling.Table_predcition import CrossTagPredictionModel

from DualHGNN.DataProcess.Label_to_Table import Label_to_Table


#加载数据
data_path = "E:/PythonProject2/DualHGNN/triplet_data/14res/train.txt"

#Tool  for
nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')

#将数据转化为标签处理
label_map_diag = {'N': 0, 'B-A': 1, 'I-A': 2, 'B-O': 3, 'I-O': 4}
label_map_sentiment = {'POS': 0, 'NEU': 1, 'NEG': 2}
label_map_other = {'N': 6, 'A': 7, 'O': 8}
# 将标签转化为索引
def convert_labels_to_indices(true_label, label_map_diag, label_map_sentiment, label_map_other):
    n = true_label.shape[0]
    true_label_indices = []
    for i in range(n):
        row_indices = []
        for j in range(n):
            if i == j:  # 对角线元素
                row_indices.append(label_map_diag.get(true_label[i, j], 0))  # 默认为 'N'
            else:  # 非对角线元素
                if true_label[i, j] in label_map_sentiment:  # 情感标签
                    row_indices.append(label_map_sentiment[true_label[i, j]])
                else:  # 其他非对角线标签
                    row_indices.append(label_map_other.get(true_label[i, j], 0))  # 默认为 'N'
        true_label_indices.append(row_indices)
    return np.array(true_label_indices)

#获取情感位置标签
def get_sentiment_positions(true_label_indices, sentiment_labels_map, sentence_len):
    # 用于存储情感在表格中的坐标位置
    sentiment_positions = []
    sentiment_label = []
    # 遍历上三角区域的所有元素（不包括对角线）
    for i in range(sentence_len):
        for j in range(i + 1, sentence_len):  # 只遍历上三角区域
            # 如果当前位置是情感标签
            if true_label_indices[i, j] in sentiment_labels_map.values():
                # 找到情感在表格中的位置，存储在列表中
                sentiment_positions.append((i, j))
                sentiment_label.append(true_label_indices[i, j])
    return sentiment_positions, sentiment_label

#Aspect-Opinion Pair Extraction 准确率
def compute_aspect_opinion_accuracy(predicted_labels, true_labels, sentence_len):
    correct = 0
    total = 0
    # 遍历对角线上的元素
    for i in range(sentence_len):
        if true_labels[i][i] == predicted_labels[i][i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

#Sentiment Polarity Classification 准确率
def compute_sentiment_accuracy(sentiment_prediction, sentiment_label_tensor):
    # sentiment_prediction 是模型的预测结果 (n, 3)
    predicted_sentiment = torch.argmax(sentiment_prediction, dim=1)
    correct = torch.sum(predicted_sentiment == sentiment_label_tensor).item()
    total = sentiment_label_tensor.size(0)
    return correct / total if total > 0 else 0




#model initial
#Syntax HGNN initialization and training
in_channels = 768
hidden_channels = 512
out_channels =  512
syn_model = Syntax_HeteroSAGE(in_channels, hidden_channels, out_channels)

#Struct_HGNN initialization and training
in_channels = 768
hidden_channels = 512
out_channels =  512
struct_model = Struct_HeteroSAGE(in_channels, hidden_channels, out_channels)

# Cross Attention initialization
hidden_dim = 512
num_heads = 8
cross_attention_module = CrossAttentionModule(hidden_dim, num_heads)

#损失函数
linear1 = nn.Linear(1024, 512)  # 第一次线性映射: 从 1024 映射到 512
linear2 = nn.Linear(512, 5)  # 第二次线性映射: 从 512 映射到 5
# Step 2: 定义两次线性映射
layer1 = nn.Linear(1024, 512)  # 第一次线性映射: 从 1024 映射到 512
layer2 = nn.Linear(512, 3)  # 第二次线性映射: 从 512 映射到 5

pool = nn.AdaptiveMaxPool1d(1)  # 最大池化层

#损失函数
criterion = CrossEntropyLoss()
optimizer = optim.Adam(
    list(syn_model.parameters()) +
    list(struct_model.parameters()) +
    list(cross_attention_module.parameters()) +
    list(linear1.parameters()) +
    list(linear2.parameters()) +
    list(layer1.parameters()) +
    list(layer2.parameters()),
    lr=0.001
)

if __name__ == "__main__":
    num_epochs = 50
    batch_size = 32

    for epoch in range(num_epochs):
        # dataset for training
        Sentence_list = Dataset_Sentence(data_path)
        true_table_list = Label_to_Table(data_path)

        # 在每个 Epoch 开始时，初始化累计损失和准确率
        total_aspect_opinion_loss = 0.0
        total_sentiment_loss = 0.0
        total_loss = 0.0
        total_aspect_opinion_correct = 0
        total_aspect_opinion_total = 0
        total_sentiment_correct = 0
        total_sentiment_total = 0

        for sentence, true_label in zip(Sentence_list,true_table_list) :
            # try:
                # 清零梯度
                optimizer.zero_grad()
                print(sentence)

                # extracting syntactic related features
                dep_features, pos_features, distance_features = BiAffine(sentence)
                g, x_dict, edge_index_dict = Syntactic_to_heterogeneous(sentence, dep_features, pos_features,distance_features)
                syn_feature_output = syn_model(x_dict, edge_index_dict)

                # structural heterogeneous Graph learning
                constituent_tree = nlp.parse(sentence)
                constituent_tree = parse_to_tuple(constituent_tree)
                struct_to_struct, struct_to_word, struct_mapping = parse_tree_to_edges(constituent_tree)
                distance_matrix = compute_distance_matrix(constituent_tree)
                position_matrix = compute_position_distance_matrix(sentence)

                # Constructing structural heterogeneous
                g, x_dict, edge_index_dict = Struct_to_heterogeneous(sentence, constituent_tree, distance_matrix, position_matrix)
                struct_feature_output = struct_model(x_dict, edge_index_dict)

                # cross Attention block
                output1 = cross_attention_module(syn_feature_output, struct_feature_output)
                output2 = cross_attention_module(struct_feature_output, syn_feature_output)
                linguistic_output = torch.cat((output1, output2), dim=1)

                # table filling and prediction
                table_prediction_module = CrossTagPredictionModel(hidden_dim=linguistic_output.size(1),sentence_len=linguistic_output.size(0))
                predicted_labels, correlation_matrix = table_prediction_module(linguistic_output)
                #predicted_labels为预测标签矩阵

                #true table label and indices
                true_label_indices = convert_labels_to_indices(true_label, label_map_diag, label_map_sentiment, label_map_other)
                # print(true_label_indices)

                #对角线损失函数计算，5分类任务{'N': 0, 'B-A': 1, 'I-A': 2, 'B-O': 3, 'I-O': 4}
                # Step 1: 提取对角线元素 (n, 1024)
                diag_elements = torch.stack([correlation_matrix[i, i] for i in range(correlation_matrix.shape[0])])  # (n, 1024)
                # Step 2: 进行第2次线性映射 (n, 512)
                mapped_diag_1 = linear1(diag_elements)  # (n, 512)
                prediction_output = linear2(mapped_diag_1)  # (n, 5)
                #标签索引斜对角线，作为真实标签值
                diag_true_labels = np.diag(true_label_indices)
                true_label = torch.tensor(diag_true_labels).long()
                #对角线损失函数的计算
                aspect_opinion_loss = criterion(prediction_output, true_label)
                # print(aspect_opinion_loss)

                #情感位置标签预测，3分类任务{'POS': 0, 'NEU': 1, 'NEG': 2}
                # Step 1: 获取情感位置坐标信息
                sentence_len = true_label_indices.shape[0]
                sentiment_positions, sentiment_label = get_sentiment_positions(true_label_indices, label_map_sentiment, sentence_len)

                #判别情感标签有没有问题
                if not sentiment_positions:
                    # 如果没有情感位置坐标，跳出当前循环
                    print("没有找到情感位置坐标，跳出当前循环")
                else:
                    print("OK")

                sentiment_label_tensor = torch.tensor(sentiment_label).long()
                sentiment_outputs = []
                for (i, j) in sentiment_positions:
                        # 获取 (i, i) 和 (j, j) 的张量
                        aspect_ii = correlation_matrix[i, i]  # (1024,)
                        opinion_jj = correlation_matrix[j, j]  # (1024,)
                        sentiment_ij = correlation_matrix[i, j]  # (1024,)
                        combined_tensor = torch.cat([aspect_ii.unsqueeze(0), opinion_jj.unsqueeze(0), sentiment_ij.unsqueeze(0)],dim=0)  # (3, 1024)
                        pooled_tensor = pool(combined_tensor.permute(1, 0)).squeeze(dim=-1).unsqueeze(0)  # (1, 1024)
                        mapped_diag_1 = layer1(pooled_tensor)  # (n, 512)
                        prediction_output2 = layer2(mapped_diag_1)  # (n, 3)
                        sentiment_outputs.append(prediction_output2)

                # 将所有的预测张量堆叠成一个张量
                sentiment_prediction = torch.cat(sentiment_outputs, dim=0)  # (n, 3)
                #情感损失计算损失
                sentiment_loss = criterion(sentiment_prediction, sentiment_label_tensor)
                #模型的最终损失
                batch_total_loss = 0.5 * aspect_opinion_loss + 0.5 * sentiment_loss

                #记录三种损失
                total_aspect_opinion_loss += aspect_opinion_loss.item()
                total_sentiment_loss += sentiment_loss.item()
                total_loss += batch_total_loss.item()

                # 计算 aspect-opinion 准确率
                predicted_array_labels = np.array(predicted_labels)
                predicted_labels_indes = convert_labels_to_indices(predicted_array_labels, label_map_diag, label_map_sentiment, label_map_other)
                aspect_opinion_acc = compute_aspect_opinion_accuracy(predicted_labels_indes, true_label_indices,sentence_len)

                # 计算情感分类准确率
                sentiment_acc = compute_sentiment_accuracy(sentiment_prediction, sentiment_label_tensor)
                # 加权平均整体准确率
                overall_accuracy = 0.5 * aspect_opinion_acc + 0.5 * sentiment_acc

                #记录三种准确率
                total_aspect_opinion_correct += aspect_opinion_acc * len(predicted_labels)
                total_aspect_opinion_total += len(predicted_labels)
                total_sentiment_correct += sentiment_acc * len(sentiment_label_tensor)
                total_sentiment_total += len(sentiment_label_tensor)

                batch_total_loss.backward()
                optimizer.step()

        # 计算整个 epoch 的平均损失和准确率
        average_aspect_opinion_loss = total_aspect_opinion_loss / len(Sentence_list)
        average_sentiment_loss = total_sentiment_loss / len(Sentence_list)
        average_loss = total_loss / len(Sentence_list)
        average_aspect_opinion_acc = total_aspect_opinion_correct / total_aspect_opinion_total
        average_sentiment_acc = total_sentiment_correct / total_sentiment_total
        average_acc = 0.5 * average_aspect_opinion_acc + 0.5 * average_sentiment_acc

        # 打开文件，写入训练结果
        with open('D2_14res_GAT.txt', 'a') as f:
          # 在每个 Epoch 结束时记录损失和准确率
          f.write(f"Epoch {epoch + 1}/{num_epochs}:\n")
          f.write(f"Aspect-Opinion Loss: {average_aspect_opinion_loss:.4f},Sentiment Loss: {average_sentiment_loss:.4f},Total Loss: {average_loss:.4f}\n")
          f.write(f"Aspect-Opinion Accuracy: {average_aspect_opinion_acc:.4f}, Sentiment Accuracy: {average_sentiment_acc:.4f}, Overall Accuracy: {average_acc:.4f}\n")

            # except Exception as e:
            #     print(f"Error processing sentence: {sentence}, Error: {e}")
            #     continue