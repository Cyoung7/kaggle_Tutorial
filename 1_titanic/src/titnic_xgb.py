# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import pandas as pd
import xgboost as xgb
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


# build xgboost model
gbm = xgb.XGBClassifier(max_depth=3,
                        n_estimators=300,
                        learning_rate=0.05)
gbm.fit(x_trained, y_trained)
y_pred = gbm.predict(x_tested)
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})

submission.to_csv('../data/submission_titanic_xgb.csv', index=False)
print(submission.shape)

from sklearn.externals import joblib
joblib.dump(gbm,'../model/titanic_xgb.model')
print('hello')
