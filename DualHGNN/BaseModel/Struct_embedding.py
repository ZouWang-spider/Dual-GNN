import torch
import itertools
import torch.nn as nn

##################  Dep_feature to Embedding  ########################
constitune_feature_labels = ['ROOT','IP','NP','VP','PU','LCP','PP','CP','DNP','ADVP','ADJP','DP','QP','NN','NR','NT','PN','VV','VC','CC','VE','VA','AS','VRD','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','PDT','POS','PRP','RB','RBR','RBS','RP','SYM','TO','WDT','WP','WP$','WRB']
dep_feature_map = {label: idx for idx, label in enumerate(constitune_feature_labels)}
# 2. 创建嵌入表
embedding_dim = 768
constitune_feature_embeddings = nn.Embedding(num_embeddings=len(constitune_feature_labels), embedding_dim=embedding_dim)
# 3. Dep_feature to Embedding
def get_level_feature_embedding(constitune_feature):
    idx = dep_feature_map.get(constitune_feature, None)
    if idx is not None:
        return constitune_feature_embeddings(torch.tensor(idx))
    else:
        return torch.zeros(embedding_dim)  # 或者其他默认值


##################  contitune_distance_feature to embedding  ########################
def get_distance_embedding(distance_map , distance):
    # 3. 为每个距离标签创建嵌入表
    distance_embedding = nn.Embedding(num_embeddings=len(distance_map), embedding_dim=embedding_dim)
    idx = distance_map.get(distance, None)
    if idx is not None:
        return distance_embedding(torch.tensor(idx))
    else:
        return torch.zeros(embedding_dim)  # 若距离值不存在，返回零向量

##################  position_feature to embedding  ########################
def get_position_embedding(distance_map , distance):
    # 3. 为每个距离标签创建嵌入表
    distance_embedding = nn.Embedding(num_embeddings=len(distance_map), embedding_dim=embedding_dim)
    idx = distance_map.get(distance, None)
    if idx is not None:
        return distance_embedding(torch.tensor(idx))
    else:
        return torch.zeros(embedding_dim)  # 若距离值不存在，返回零向量