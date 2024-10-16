import nltk
import dgl
import re
import torch
import networkx as nx
from supar import Parser
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel

nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')
parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
#使用BiAffine对句子进行处理得到arcs、rels、probs
def BiAffine(sentence):
      text = sentence.split()
      # print(text)

      ###################POS feature#####################
      ann = nlp.pos_tag(sentence)
      pos_tags = [pair[1] for pair in ann]
      # print(pos_tags)

      # 初始化空的pos_features矩阵
      pos_features = []
      # 构建pos_features矩阵
      for i in range(len(pos_tags)):
            row = []
            for j in range(len(pos_tags)):
                  if i == j:
                        row.append('self')
                  else:
                        row.append(f'{pos_tags[i]}-{pos_tags[j]}')
            pos_features.append(row)

      ####################Depency feature#####################
      dataset = parser.predict([text], prob=True, verbose=True)

      #dependency feature
      rels = dataset.rels[0]
      # print(f"arcs:  {dataset.arcs[0]}\n"
      #       f"rels:  {dataset.rels[0]}\n")

      # 构建句子的图，由弧-->节点
      arcs = dataset.arcs[0]  # 边的信息
      edges = [i + 1 for i in range(len(arcs))]
      for i in range(len(arcs)):
            if arcs[i] == 0:
                  arcs[i] = edges[i]

      # 将节点的序号减一，以便适应DGL graph从0序号开始
      arcs = [arc - 1 for arc in arcs]
      edges = [edge - 1 for edge in edges]
      graph = (arcs, edges)
      # print(rels)
      # print("graph:", graph)

      # 节点的数量
      n = len(graph[0])
      # 初始化dep_features矩阵
      dep_features = [['-' for _ in range(n)] for _ in range(n)]
      # 填充对角线的'self'
      for i in range(n):
            dep_features[i][i] = 'self'
      # 填充dep_features矩阵，处理正向和反向边
      for i, (src, tgt) in enumerate(zip(graph[0], graph[1])):
            dep_features[src][tgt] = rels[i]
            dep_features[tgt][src] = rels[i]  # 处理反向边

      #####################Depency tree distance#####################
      # 创建无向图,使用图遍历来获取节点之间的距离
      G = nx.Graph()
      # 使用zip函数将两个列表配对，生成边的列表
      edges = list(zip(graph[0], graph[1]))
      G.add_edges_from(edges)
      # 获取图中所有节点
      nodes = sorted(G.nodes())
      # 初始化distance_features矩阵
      n = len(nodes)
      distance_features = [[0] * n for _ in range(n)]
      # 计算所有节点对之间的最短路径距离
      for i, src in enumerate(nodes):
            for j, tgt in enumerate(nodes):
                  if i != j:
                        # 使用网络X的shortest_path_length来计算节点之间的距离
                        distance_features[i][j] = nx.shortest_path_length(G, source=src, target=tgt)

      return dep_features, pos_features, distance_features


# 加载BERT模型和分词器
model_name = 'E:/bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
#节点特征
def BERT_Embedding(sentence):
      # text = sentence.split()
      # 正则表达式匹配由连字符连接的多个单词，并在连字符两边添加空格
      modified_sentence = re.sub(r'(\w+)-(\w+)(-\w+)*', lambda m: ' - '.join(re.split('-', m.group())), sentence)
      # 将句子分词
      text = modified_sentence.split()
      # 获取单词节点特征
      marked_text1 = ["[CLS]"] + text + ["[SEP]"]

      # 将分词转化为词向量
      input_ids1 = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs1 = model(input_ids1)
      # 获取词向量
      word_embeddings = outputs1.last_hidden_state
      # 提取单词对应的词向量（去掉特殊标记的部分）
      word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      # 使用切片操作去除第一个和最后一个元素
      word_feature = word_embeddings[0][1:-1, :]  # 单词特征
      return word_feature


# sentence ='This fried chicken was terrible, but the pizza was delicious'
# dep_features, pos_features, distance_features = BiAffine(sentence)
# print("依存特征矩阵", dep_features)
# print('词性特征矩阵', pos_features)
# print("依存树距离特征", distance_features)

# word_feature = BERT_Embedding(sentence)
# print(word_feature.shape)  #(6,768)
# print(word_feature)

