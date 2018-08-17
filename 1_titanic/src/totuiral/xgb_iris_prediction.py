# -*-coding:utf-8-*-
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
import random

iris = datasets.load_iris()
x = iris.data
y = iris.target
# ndarray or DataFrame均可
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=random.randint(0, 999))
# ndarray;DataFrame;svm file;binary均可
train_dst = xgb.DMatrix(x_train, label=y_train)
valid_dst = xgb.DMatrix(x_valid, label=y_valid)

dump_svmlight_file(x_train, y_train, '../../data/iris_dtrain.svm', zero_based=True)
dump_svmlight_file(x_valid, y_valid, '../../data/iris_dvalid.svm', zero_based=True)

x_train_svm = xgb.DMatrix('../../data/iris_dtrain.svm')
x_valid_svm = xgb.DMatrix('../../data/iris_dvalid.svm')

params = {
    'max_depth': 3,
    'eta': 0.3,
    'silent': 1,
    'objective': 'multi:softprob',
    'num_class': 3
}
num_round = 20

bst = xgb.train(params=params, dtrain=x_train_svm, num_boost_round=num_round)
preds = bst.predict(x_valid_svm)
best_preds = np.asarray(np.argmax(preds, axis=1))
print("Numpy array precision:", precision_score(y_valid, best_preds, average='macro'))

# 模型保存
bst.dump_model('../../model/iris_xgb.model')
joblib.dump(bst, '../../model/iris_xgb.pkl', compress=True)

print('hello')
