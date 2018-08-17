# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

sb.set(style='whitegrid')

# loading and view data set
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


# print(train_data.info())
# print(train_data.describe())
# print(train_data.head())
# print(train_data.keys())
# print(test_data.keys())

# deal with NaN value
def null_table(train, test):
    print('train data frame')
    print(pd.isnull(train).sum(axis=0))
    print('test data frame')
    print(pd.isnull(test).sum(axis=0))


null_table(train_data, test_data)
train_data.drop(labels=['Cabin', 'Ticket'], axis=1, inplace=True)
test_data.drop(labels=['Cabin', 'Ticket'], axis=1, inplace=True)
null_table(train_data, test_data)

train_copy = train_data.copy()
train_copy.dropna(inplace=True)
# print(train_data.info())
# print(train_copy.info())
# sb.distplot(train_copy['Age'])
# plt.show()

# fill NaN
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
# null_table(train_data, test_data)

# plotting and visualizing data
# to see the entropy and information gain of the feature
# sb.barplot(x='Sex', y='Survived', data=train_data)
# plt.title('Distribution of Survival based on Gender')
# plt.show()

# Sex
sur_females = train_data[train_data['Sex'] == 'female']['Survived'].sum()
sur_males = train_data[train_data['Sex'] == 'male']['Survived'].sum()
print('Proportion of Females/Males who survived:')
total_sur = sur_females + sur_males
print('females:', sur_females / total_sur, 'male:', sur_males / total_sur)
# sex is a good feature


# Class
# sb.barplot(x='Pclass',y='Survived',data=train_data)
# plt.ylabel('Survived Rate')
# plt.title("Distribution of Survival Based on Class")
# plt.show()
# sb.barplot(x='Pclass', y='Survived', hue='Sex', data=train_data)
# plt.ylabel("Survival Rate")
# plt.title("Survival Rates Based on Gender and Class")
# plt.show()
#
# sb.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data)
# plt.ylabel("Survival Rate")
# plt.title("Survival Rates Based on Gender and Class")
# plt.show()

# Age
sur_ages = train_data[train_data['Survived'] == 1]['Age']
no_sur_ages = train_data[train_data['Survived'] == 0]['Age']
plt.subplot(1, 2, 1)
# 绘制单变量的观测分布
# sb.distplot(sur_ages, kde=False)
# plt.axis([0, 100, 0, 100])
# plt.title("Survived")
# plt.ylabel("Proportion")
# plt.subplot(1, 2, 2)
# sb.distplot(no_sur_ages, kde=False)
# plt.axis([0, 100, 0, 100])
# plt.title("Didn't Survive")
# plt.show()

# 从途中看出，更年轻的人存活的纪律更大
# sb.stripplot(x='Survived',y='Age',data=train_data,jitter=True)
# plt.show()

# 查看所有特征之间的关系
# sb.pairplot(train_data)
# plt.show()

# feature engineering
# print(train_data.sample(5))
sex_mapping = {'male': 0, 'female': 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping).astype(int)
test_data['Sex'] = test_data['Sex'].map(sex_mapping).astype(int)

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping).astype(int)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping).astype(int)
# print(train_data.sample(5))

# add a new feature
train_data['FamSize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamSize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data['IsAlone'] = train_data['FamSize'].apply(lambda x: 1 if x == 1 else 0)
test_data['IsAlone'] = test_data['FamSize'].apply(lambda x: 1 if x == 1 else 0)

train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
test_data['Title'] = test_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
title_replacements = {'Mlle': 'Other', 'Major': 'Other', 'Col': 'Other', 'Sir': 'Other', 'Don': 'Other', 'Mme': 'Other',
                      'Jonkheer': 'Other', 'Lady': 'Other', 'Capt': 'Other', 'Countess': 'Other', 'Ms': 'Other',
                      'Dona': 'Other', 'Rev': 'Other', 'Dr': 'Other'}
train_data.replace({'Title': title_replacements}, inplace=True)
test_data.replace({'Title': title_replacements}, inplace=True)
title_mapping = {'Miss': 0, 'Mr': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}
train_data['Title'] = train_data['Title'].map(title_mapping).astype(int)
test_data['Title'] = test_data['Title'].map(title_mapping).astype(int)

print(set(train_data['Title']))
print(train_data.sample(5))
print('hello')

# Model Fitting and Predicting
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# define feature in training/Test set
features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize", "IsAlone", "Title"]
x_trained = train_data[features]
y_trained = train_data['Survived']
x_tested = test_data[features]

# validate data set
from sklearn.model_selection import train_test_split
import random

x_train, x_valid, y_train, y_valid = train_test_split(x_trained, y_trained, test_size=0.2,
                                                      random_state=random.randint(0, 999))

# SVC model
# svc_cls = SVC(kernel='linear', gamma=3)
# svc_cls.fit(x_train, y_train)
# pred_svc = svc_cls.predict(x_valid)
# acc_svc = accuracy_score(y_valid, pred_svc)
# 0.8268156424581006
# print(acc_svc)

# linearSVC

# linsvc_cls = LinearSVC()
# linsvc_cls.fit(x_train,y_train)
# pred_linsvc= linsvc_cls.predict(x_valid)
# acc_linsvc = accuracy_score(y_valid,pred_linsvc)
# 0.8324022346368715
# print(acc_linsvc)

# random Forest Model
# rf_cls = RandomForestClassifier()
# rf_cls.fit(x_train,y_train)
# pred_rf = rf_cls.predict(x_valid)
# acc_rf = accuracy_score(y_valid,pred_rf)
# 0.8156424581005587
# print(acc_rf)

# logistic regression model
# logreg_cls = LogisticRegression()
# logreg_cls.fit(x_train,y_train)
# pred_log = logreg_cls.predict(x_valid)
# acc_log = accuracy_score(y_valid,pred_log)
# 0.8100558659217877
# print(acc_log)

# kNeighbors model
# knn_cls = KNeighborsClassifier()
# knn_cls.fit(x_train,y_train)
# pred_knn = knn_cls.predict(x_valid)
# acc_knn = accuracy_score(y_valid,pred_knn)
# 0.7150837988826816
# print(acc_knn)


# GaussianNB model
# gnb_cls = GaussianNB()
# gnb_cls.fit(x_train,y_train)
# pred_gnb = gnb_cls.predict(x_valid)
# acc_gnb = accuracy_score(y_valid,pred_gnb)
# 0.8044692737430168
# print(acc_gnb)

# decision tree
# dt_cls = DecisionTreeClassifier()
# dt_cls.fit(x_train,y_train)
# pred_dt = dt_cls.predict(x_valid)
# acc_dt = accuracy_score(y_valid,pred_dt)
# 0.8100558659217877
# print(acc_dt)

# 找到表现最好的模型
rf_cls = RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt', 'auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
              }

acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(rf_cls, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_trained, y_trained)

rf_clf = grid_obj.best_estimator_

rf_clf.fit(x_train, y_train)

predictions = rf_clf.predict(x_valid)
print(accuracy_score(y_valid, predictions))

# submission
rf_clf.fit(x_trained, y_trained)
sub_pred = rf_clf.predict(x_tested)
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                           'Survived': sub_pred})
submission.to_csv('../data/submission_titanic.csv', index=False)
print(submission.shape)

# 模型保存，持久化
from sklearn.externals import joblib
joblib.dump(rf_clf, "../model/titanic_rf.model")
