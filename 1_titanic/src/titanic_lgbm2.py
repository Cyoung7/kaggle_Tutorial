# The usuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regular expressions (urrggghhhhhhhhh) for handling string features
import re

# LightGBM
import lightgbm as lgb

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV

# Import both data sets
trainRaw = pd.read_csv('../data/train.csv')
testRaw = pd.read_csv('../data/test.csv')

# And concatonate together
nTrain = trainRaw.shape[0]
full = pd.concat([trainRaw, testRaw], axis=0)


# %% Cabins

def ADSplit(s):
    """
    Function to try and extract cabin letter and number from the cabin column.
    Runs a regular expression that finds letters and numbers in the
    string. These are held in match.group, if they exist.
    """

    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)

    try:
        letter = match.group(1)
    except:
        letter = ''

    try:
        number = match.group(2)
    except:
        number = 9999

    return letter, number


def DR(s):
    """
    From the cabin string, try and extract letter, number, and number of cabins
    """

    # Check contents
    if isinstance(s, (int, float)):
        # If field is empty, return nothing
        letter = ''
        number = ''
        nRooms = 9999
    else:
        # If field isn't empty, split sting on space. Some strings contain
        # multiple cabins.
        s = s.split(' ')
        # Count the cabins based on number of splits
        nRooms = len(s)
        # Just take first cabin for letter/number extraction
        s = s[0]

        letter, number = ADSplit(s)

    return [letter, number, nRooms]


# Apply DR function to each cell in Cabin column using pandas apply method.
out = full['Cabin'].apply(DR)
# Outout tuple with 3 values for each row, convert this to pandas df
out = out.apply(pd.Series)
# And name the columns
out.columns = ['CL', 'CN', 'nC']

# Then concatenate these columns to the dataset
full = pd.concat([full, out], axis=1)

# %% Family

# Add some family features directly to new columns in the dataset
# Size
full['fSize'] = full['SibSp'] + full['Parch'] + 1
# Ratio
full['fRatio'] = (full['Parch'] + 1) / (full['SibSp'] + 1)
# Adult?
full['Adult'] = full['Age'] > 18

# %% Names
# Extract titles from Name column, standardise

titleDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Sir",
    "Don": "Sir",
    "Sir": "Sir",
    "Dr": "Dr",
    "Rev": "Rev",
    "theCountess": "Lady",
    "Dona": "Lady",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Lady"
}


def splitName(s, titleDict):
    """
    Extract title from name, replace with value in title dictionary. Also
    return surname.
    """

    # Remove '.' from name string
    s = s.replace('.', '')
    # Split on spaces
    s = s.split(' ')
    # get surname
    surname = s[0]

    # Get title - loop over titleDict, if s matches a key, take the
    # corresponding value as the title
    title = [t for k, t in titleDict.items() if str(k) in s]

    # If no matching keys in title dict, use 'Other'.
    if title == []:
        title = 'Other'
    else:
        # Title is a list, so extract contents
        title = title[0]

    # Return surname (stripping remaining ',') and title as string
    return surname.strip(','), title


# Apply functions to df and concatenate new columns as before
out = full['Name'].apply(splitName, args=[titleDict])
out = out.apply(pd.Series)
out.columns = ['Surname', 'Title']

full = pd.concat([full, out], axis=1)

# %% Categorical columns

# List of categorical columns to recode
catCols = ['Sex', 'Embarked', 'CL', 'CN', 'Surname', 'Title']

# Recode
for c in catCols:
    # Convert column to pd.Categotical
    full[c] = pd.Categorical(full[c])
    # Extract the cat.codes and replace the column with these
    full[c] = full[c].cat.codes
    # Convert the cat codes to categotical...
    full[c] = pd.Categorical(full[c])

# Generate a logical index of categorical columns to maybe use with LightGBM later
catCols = [i for i, v in enumerate(full.dtypes) if str(v) == 'category']

# %% Age

# Replace missing age values with median.
# See ither kernels for more sophisticated ways of doing this!
full.loc[full.Age.isnull(), 'Age'] = np.median(full['Age'].loc[full.Age.notnull()])

# %% Split datasets

train = full.iloc[0:nTrain, :]
test = full.iloc[nTrain::, :]


def prepLGB(data, classCol='', IDCol='', fDrop=[]):
    # Drop class column
    if classCol != '':
        labels = data[classCol]
        fDrop = fDrop + [classCol]
    else:
        labels = []

    if IDCol != '':
        IDs = data[IDCol]
        fDrop = fDrop + [IDCol]
    else:
        IDs = []

    if fDrop != []:
        data = data.drop(fDrop, axis=1)

    # Create LGB mats
    lData = lgb.Dataset(data, label=labels, free_raw_data=False,
                        feature_name=list(data.columns),
                        categorical_feature='auto')

    return lData, labels, IDs, data


# Specify columns to drop
fDrop = ['Ticket', 'Cabin', 'Name']

# Split training data in to training and validation sets.
# Validation set is used for early stopping.
trainData, validData = train_test_split(train, test_size=0.4)

# Prepare the data sets
trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
                                                       classCol='Survived',
                                                       IDCol='PassengerId',
                                                       fDrop=fDrop)

validDataL, validLabels, validIDs, validData = prepLGB(validData,
                                                       classCol='Survived',
                                                       IDCol='PassengerId',
                                                       fDrop=fDrop)

testDataL, _, _, testData = prepLGB(test,
                                    classCol='Survived',
                                    IDCol='PassengerId',
                                    fDrop=fDrop)

# Set params
# Scores ~0.784
params = {'boosting_type': 'gbdt',
          'max_depth': -1,
          'objective': 'binary',
          'nthread': 5,  # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class': 1,
          'metric': 'binary_error',
          'verbose':-1}

# Train model
# gbm = lgb.train(params,
#                 trainDataL,
#                 100000,  # Max number of trees
#                 valid_sets=[trainDataL, validDataL],
#                 early_stopping_rounds=50,
#                 verbose_eval=4)
#
# # Run prediction on training, validation, and test sets using raw data
# # Specify which model iteration to use
# predsValid = gbm.predict(validData, num_iteration=gbm.best_iteration)
# predsTrain = gbm.predict(trainData, num_iteration=gbm.best_iteration)
# predsTest = gbm.predict(testData, num_iteration=gbm.best_iteration)


def assessMod(predsTrain, yTrain, predsValid=[], yValid=[],
              report=True, plot=True):
    """
    Using sklearn functions, return accuracy and ROC metrics (tpr, fpr, auc)
    for training data and validation data (if included).

    preds should be model preditions, yTrain and yValid should be labels.

    """
    trainAcc = accuracy_score(yTrain, np.round(predsTrain))
    fprTrain, tprTrain, thresholdsTrain = roc_curve(yTrain, predsTrain)
    trainAUC = auc(fprTrain, tprTrain)

    if predsValid != []:
        accuracy_score(yValid, np.round(predsValid))
        fprValid, tprValid, thresholdsValid = roc_curve(yValid, predsValid)
        validAcc = accuracy_score(yValid, np.round(predsValid))
        validAUC = auc(fprValid, tprValid)
    else:
        validAcc = np.nan
        fprValid = np.nan
        tprValid = np.nan
        validAUC = np.nan

    if report:
        print('Train accuracy:', trainAcc, '| Train AUC:',
              trainAUC)
        if not isinstance(predsValid, list):
            print('Validation accuracy:', validAcc, '| Test AUC:',
                  validAUC)

        print('-' * 30)

    # Plot
    if plot:
        plotROC(tprTrain, fprTrain, label='Train')
        if not isinstance(predsValid, list):
            plotROC(tprValid, fprValid, label='Valid')

    # Stats output
    stats = {'fprTrain': fprTrain,
             'fprValid': fprValid,
             'tprTrain': tprTrain,
             'tprValid': tprValid,
             'trainAcc': trainAcc,
             'validAcc': validAcc,
             'trainAUC': trainAUC,
             'validAUC': validAUC}

    return stats


def plotROC(tpr, fpr, label=''):
    """
    Plot ROC curve from tpr and fpr.
    """
    plt.plot(fpr, tpr, label=label)
    plt.legend()
    plt.ylabel('True positive rate.')
    plt.xlabel('False positive rate')
    plt.show()


# Report model performance on training and validation sets
# assessMod(predsTrain, trainLabels,
#           predsValid=predsValid, yValid=validLabels,
#           report=True, plot=False)
#
# # Plot importance
# lgb.plot_importance(gbm, importance_type="split", title="split")
# plt.show()
#
# lgb.plot_importance(gbm, importance_type="gain", title='gain')
# plt.show()
#
# # Importance values are also available in:
# print(gbm.feature_importance("split"))
# print(gbm.feature_importance("gain"))
#
# # Save submission
# sub = pd.DataFrame()
# sub['PassengerId'] = test['PassengerId']
# sub['Survived'] = np.int32(predsTest > 0.5)
# sub.to_csv('sub1.csv', index=False)

# Prepare data set using all the training data
allTrainDataL, allTrainLabels, _, allTrainData = prepLGB(train,
                                                         classCol='Survived',
                                                         IDCol='PassengerId',
                                                         fDrop=fDrop)
# Create parameters to search
gridParams = {
    'learning_rate': [0.01],
    'n_estimators': [24, 48],
    'num_leaves': [6, 12, 20, 30],
    'boosting_type': ['gbdt'],
    'objective': ['binary'],
    'seed': [500],
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.7, 1],
    'reg_alpha': [1],
    'reg_lambda': [0, 1],
}

# Create classifier to use. Note that parameters have to be input manually
# not as a dict!
mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         n_jobs=3,  # Updated from 'nthread'
                         silent=True,
                         max_depth=params['max_depth'],
                         max_bin=params['max_bin'],
                         subsample_for_bin=params['subsample_for_bin'],
                         subsample=params['subsample'],
                         subsample_freq=params['subsample_freq'],
                         min_split_gain=params['min_split_gain'],
                         min_child_weight=params['min_child_weight'],
                         min_child_samples=params['min_child_samples'],
                         scale_pos_weight=params['scale_pos_weight'],
                         verbose=-1)

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams,verbose=1,cv=5,n_jobs=-1)
# Run the grid
grid.fit(allTrainData, allTrainLabels)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

# Train
gbm = lgb.train(params,
                trainDataL,
                100000,
                valid_sets=[trainDataL, validDataL],
                early_stopping_rounds=50,
                verbose_eval=4)

# Plot importance
lgb.plot_importance(gbm)
plt.show()

# Predict
predsValid = gbm.predict(validData, num_iteration=gbm.best_iteration)
predsTrain = gbm.predict(trainData, num_iteration=gbm.best_iteration)
predsTest = gbm.predict(testData, num_iteration=gbm.best_iteration)

# Print assessment
assessMod(predsTrain, trainLabels, predsValid=predsValid, yValid=validLabels,
          report=True, plot=True)

# Save submission
sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = np.int32(predsTest > 0.5)
sub.to_csv('../data/submission_titanic_lgb2.csv', index=False)
