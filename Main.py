import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# reader = pd.read_csv("train.csv")
# print(reader)

with open("train.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
# print(rows)

stopwords_english = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                     'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
                     'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from',
                     'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
                     'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                     'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
                     'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
                     'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                     'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
                     'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                     'further', 'was', 'here', 'than'}

# 数据预处理
train = []
train_x = []
train_y = []
vocab_set = set()
for row in rows[1:]:
    row[1] = row[1].lower()
    row[1] = ' '.join(
        (' '.join(c if c.isalnum() and c not in stopwords_english else ' ' for c in row[1].split())).split())
    train.append(row[0:2])
    vocab_set = vocab_set | set(row[1].split())
    train_x.append(row[1])
    train_y.append(row[0])

print('数据: ', len(train))

# 词表
vocab_list = list(vocab_set)
print('词表：', len(vocab_list))

print(train_x)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_x)
word = vectorizer.get_feature_names()
print(len(X.toarray()), len(X.toarray()[0]))
# print(X.toarray())
# print(word)
train_x_vec = X.toarray()
print(train_x_vec)
print(train_x_vec.shape)
# # 词向量
# vec = []
# for row in train_x:
#     tmp = [0] * len(vocab_list)
#     for word in row.split():
#         if word in vocab_list:
#             tmp[vocab_list.index(word)] = 1
#         else:
#             print('%s is not in vocab_list.' % word)
#     vec.append(tmp)
# print('词向量：', len(vec))

# NOTE: n_components后面如果为数字不可以加引号'
pca = PCA(n_components=10, svd_solver='auto')
pca_res = pca.fit_transform(train_x_vec)
print(pca_res)

clf = GaussianNB()
clf.fit(pca_res, train_y)


test_csv = pd.read_csv('test.csv')
print(test_csv)
