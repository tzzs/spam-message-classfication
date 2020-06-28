from sklearn.naive_bayes import GaussianNB
import Main2

train_x, train_y = Main2.data_prepare('train.csv')

print(len(train_x), len(train_y))
print(train_x)
print(train_y)

# 用于分割测试集和训练集 与下面注释部分一起使用
train_test_flag = len(train_x)

clf = GaussianNB()
clf.fit(train_x[:train_test_flag], train_y[:train_test_flag])

# predict = clf.predict(train_x[train_test_flag:])
# print(predict)

# acc = accuracy_score(predict, train_y[train_test_flag:])
# print('准确率: ', acc)

test_x, id = Main2.data_prepare('test.csv')
print(test_x)
print(id)

pre = clf.predict(test_x)

with open('result.csv', 'w+') as csvfile:
    csvfile.write('SmsId,Label\n')

    for i in range(len(id)):
        csvfile.write(id[i]+','+pre[i]+'\n')

print("predict success.")


