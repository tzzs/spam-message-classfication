from sklearn import svm
from sklearn.metrics import accuracy_score

import Main2

train_x, train_y = Main2.data_prepare('train.csv')

print(len(train_x), len(train_y))
print(train_x)
print(train_y)

train_test_flag = 5000

clf = svm.SVC()
clf.fit(train_x[:train_test_flag], train_y[:train_test_flag])

predict = clf.predict(train_x[train_test_flag:])
print(predict)
acc = accuracy_score(predict, train_y[train_test_flag:])
print(acc)

test_x, id = Main2.data_prepare('test.csv')
# print(test_x)
# print(id)

pre = clf.predict(test_x)

with open('result.csv', 'w+') as csvfile:
    csvfile.write('SmsId,Label\n')

    for i in range(len(id)):
        csvfile.write(id[i]+','+pre[i]+'\n')

print("predict success.")


