import csv
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

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


def data_prepare(csv_name):
    with open(csv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # 数据预处理
    train = []
    train_x = []
    train_y = []
    for row in rows[1:]:
        row[1] = row[1].lower()
        row[1] = ' '.join(
            (' '.join(c if c.isalnum() and c not in stopwords_english else ' ' for c in row[1].split())).split())
        train.append(row[0:2])
        train_x.append(row[1])
        train_y.append(row[0])

    # 词表
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_x)

    # 词向量
    train_x_vec = X.toarray()

    # NOTE: n_components后面如果为数字不可以加引号'
    pca = PCA(n_components=100, svd_solver='auto')
    # 计算 PCA 主成分分析矩阵
    pca_res = pca.fit_transform(train_x_vec)

    return pca_res, train_y
