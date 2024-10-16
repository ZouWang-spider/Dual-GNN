from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
sentence ='The food was pretty traditional but it was hot and good with large portions.'


# Constituency Tree Visualization
import nltk
from nltk.tree import Tree
# 将构成树字符串转换为 NLTK 的 Tree 对象
nltk_tree = Tree.fromstring(nlp.parse(sentence).strip())
# 绘制树状结构
nltk_tree.draw()