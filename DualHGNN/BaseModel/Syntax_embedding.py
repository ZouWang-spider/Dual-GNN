import torch
import itertools
import torch.nn as nn

##################  Dep_feature to Embedding  ########################
dep_feature_labels = ['abbrev', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux','auxpass','cc','ccomp','complm','conj','cop',
                      'csubj','csubjpass','dep','det','dobj','expl','infmod','iobj','mark','mwe','neg','nn','npadvmod','nsubj','nsubjpass','num',
                      'number','parataxis','partmod','pcomp','pobj','poss','possessive','preconj','predet','prep','prepc','prt','punct','purpcl',
                      'quantmod','rcmod','ref','rel','root','tmod','xcomp','xsubj','subj','nsubj','top','npsubj','csubj','xsubj','obj','dobj',
                      'iobj','range','pobj','lobj','comp','ccomp','xcomp','acomp','tcomp','lccomp','mod','pass','tmod','rcmod','numod','ornmod',
                      'clf','nmod','advmod','vmod','prnmod','neg','det','poss','dvpm','dvpmod','assm','assmod','plmod','partmod','etc','conj',
                      'cop','cc','attr','cordmod','ba','tclaus','cpm']
dep_feature_map = {label: idx for idx, label in enumerate(dep_feature_labels)}
# 2. 创建嵌入表
embedding_dim = 768
dep_feature_embeddings = nn.Embedding(num_embeddings=len(dep_feature_labels), embedding_dim=embedding_dim)
# 3. Dep_feature to Embedding
def get_dep_feature_embedding(dep_feature):
    idx = dep_feature_map.get(dep_feature, None)
    if idx is not None:
        return dep_feature_embeddings(torch.tensor(idx))
    else:
        return torch.zeros(embedding_dim)  # 或者其他默认值


##################  POS_feature to embedding  ########################
pos_feauter_labels = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WH','WRB']
# 2. 使用 itertools.product 生成两两词性特征的组合
pos_feature_combinations = ['-'.join(comb) for comb in itertools.product(pos_feauter_labels, repeat=2)]
# 3. 创建词性组合的映射表
pos_feature_map = {feature: idx for idx, feature in enumerate(pos_feature_combinations)}
# 4. 创建嵌入层，定义嵌入维度
pos_feature_embedding = nn.Embedding(num_embeddings=len(pos_feature_combinations), embedding_dim=embedding_dim)

def get_pos_feature_embedding(pos_feature):
    idx = pos_feature_map.get(pos_feature, None)
    if idx is not None:
        return pos_feature_embedding(torch.tensor(idx))  # 返回对应嵌入
    else:
        return torch.zeros(embedding_dim)  # 如果词性组合不存在，返回默认零向量




##################  distance_feature to embedding  ########################
def get_distance_embedding(distance_map , distance):
    # 3. 为每个距离标签创建嵌入表
    distance_embedding = nn.Embedding(num_embeddings=len(distance_map), embedding_dim=embedding_dim)
    idx = distance_map.get(distance, None)
    if idx is not None:
        return distance_embedding(torch.tensor(idx))
    else:
        return torch.zeros(embedding_dim)  # 若距离值不存在，返回零向量