# -*-coding:utf-8-*-
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# loading and view data set
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

train_data.drop(labels=['Cabin', 'Ticket'], axis=1, inplace=True)
test_data.drop(labels=['Cabin', 'Ticket'], axis=1, inplace=True)

# fill NaN
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

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

# define feature in training/Test set
features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize", "IsAlone", "Title"]
x_trained = train_data[features]
print(x_trained.sample(5))
y_trained = train_data['Survived']
x_tested = test_data[features]

# validate data set
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import random

x_train, x_valid, y_train, y_valid = train_test_split(x_trained, y_trained, test_size=0.2,
                                                      random_state=random.randint(0, 999))

# create LGBM dataset from the test

categorical_features = ['Pclass', 'Sex', 'Embarked', 'IsAlone', 'Title']
train_dateset = lgbm.Dataset(data=x_train, label=y_train, categorical_feature=categorical_features,
                             free_raw_data=False)

test_dataset = lgbm.Dataset(data=x_valid, label=y_valid, categorical_feature=categorical_features,
                            free_raw_data=False)

trained_dataset = lgbm.Dataset(data=x_trained, label=y_trained, categorical_feature=categorical_features,
                               free_raw_data=False)

# define hyper_parameters for lgbm
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.7,
    'num_leaves': 41,
    'metric': 'binary_logloss',
    'drop_rate': 0.15
}

# train model
evaluation_results = {}
# Booster
lgbm_cls = lgbm.train(train_set=train_dateset,
                      params=lgbm_params,
                      valid_sets=[train_dateset, test_dataset],
                      valid_names=['Train', 'Test'],
                      evals_result=evaluation_results,
                      num_boost_round=10,
                      early_stopping_rounds=100,
                      verbose_eval=20
                      )
opimum_boost_rounds = lgbm_cls.best_iteration

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=[15, 4])
# Plot the log loss during training
axs[0].plot(evaluation_results['Train']['binary_logloss'], label='Train')
axs[0].plot(evaluation_results['Test']['binary_logloss'], label='Test')
axs[0].set_ylabel('Log loss')
axs[0].set_xlabel('Boosting round')
axs[0].set_title('Training performance')
axs[0].legend()

# Plot feature importance
importances = pd.DataFrame({'features': lgbm_cls.feature_name(),
                            'importance': lgbm_cls.feature_importance()}).sort_values('importance', ascending=False)
axs[1].bar(x=np.arange(len(importances)), height=importances['importance'])
axs[1].set_xticks(np.arange(len(importances)))
axs[1].set_xticklabels(importances['features'])
axs[1].set_ylabel('Feature importance (# times used to split)')
axs[1].set_title('Feature importance')

# plt.show()

preds = np.round(lgbm_cls.predict(x_valid))
print('Accuracy score = \t {}'.format(accuracy_score(y_valid, preds)))
print('Precision score = \t {}'.format(precision_score(y_valid, preds)))
print('Recall score =   \t {}'.format(recall_score(y_valid, preds)))
print('F1 score =      \t {}'.format(f1_score(y_valid, preds)))

clf_final = lgbm.train(train_set=trained_dataset,
                       params=lgbm_params,
                       num_boost_round=opimum_boost_rounds,
                       verbose_eval=0
                       )

y_pred = np.round(clf_final.predict(x_tested)).astype(int)
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})

submission.to_csv('../data/submission_titanic_lgb.csv', index=False)
print(submission.shape)
