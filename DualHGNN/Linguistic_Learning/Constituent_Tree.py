from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
sentence ='The fried chicken was terrible'
print ('Tokenize:', nlp.word_tokenize(sentence))
print ('Part of Speech:', nlp.pos_tag(sentence))
print ('Named Entities:', nlp.ner(sentence))
print ('Constituency Parsing:', nlp.parse(sentence))
print ('Dependency Parsing:', nlp.dependency_parse(sentence))

import nltk
from nltk.tree import Tree

# 将构成树字符串转换为 NLTK 的 Tree 对象
nltk_tree = Tree.fromstring(nlp.parse(sentence).strip())

# 绘制树状结构
nltk_tree.draw()
