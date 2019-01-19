'''

Title:
Machine Learning - Income Prediction

Authors:
Shishir Singapura Lakshminarayan - ssl495
Mayank Grover                    - mg5229

'''

# coding: utf-8

# In[1]:


# Read parameters in input
import sys

# Utilities
import pandas as pd
import numpy as np
from collections import Counter

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Cross-validation
from sklearn.model_selection import cross_validate

# Feature Selection
from sklearn.feature_selection import RFE


# In[2]:


# Reading parameters as files
train_file_path = 'census_train.csv' if len(sys.argv) == 1 else sys.argv[1]
test_file_path = 'census_test.csv' if len(sys.argv) == 1 else sys.argv[2]

# print(train_file_path)
# print(test_file_path)


# In[3]:


# Read CSV files

df_train = pd.read_csv(train_file_path, header = None, names = ['idnum', 'age', 'workerclass', 'interestincome', 
'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 
'sex', 'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin', 'wages'])

X_test = pd.read_csv(test_file_path, header = None, names = ['idnum', 'age', 'workerclass', 'interestincome', 
'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 
'sex', 'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin'])


# In[4]:


def clean_dataframe(df):
    
    df['age'] = df['age'].replace('?', df['age'].mean())

    df['workerclass'] = df['workerclass'].replace('?', 0)
    df['workerclass'] = df['workerclass'].replace(['1','2'], 1)
    df['workerclass'] = df['workerclass'].replace(['3','4','5'], 2)
    df['workerclass'] = df['workerclass'].replace(['6','7'], 3)
    df['workerclass'] = df['workerclass'].replace(['8'], 4)
    df['workerclass'] = df['workerclass'].replace(['9'], 5)

    df['interestincome'] = df['interestincome'].replace('?', 0)

    df['traveltimetowork'] = df['traveltimetowork'].replace('?', 0)

    df['occupancytransport'] = 0
    df['occupancytransport'][(df['vehicleoccupancy'] == '1') & (df['meansoftransport'] == '1')] = 1
    df['occupancytransport'][(df['vehicleoccupancy'] == '2') & (df['meansoftransport'] == '1')] = 2
    df['occupancytransport'][(df['vehicleoccupancy'] == '3') & (df['meansoftransport'] == '1')] = 3
    df['occupancytransport'][(df['vehicleoccupancy'] == '4') & (df['meansoftransport'] == '1')] = 4
    df['occupancytransport'][(df['vehicleoccupancy'] == '5') & (df['meansoftransport'] == '1')] = 5
    df['occupancytransport'][(df['vehicleoccupancy'] == '6') & (df['meansoftransport'] == '1')] = 6
    df['occupancytransport'][(df['vehicleoccupancy'] == '7') & (df['meansoftransport'] == '1')] = 7
    df['occupancytransport'][(df['vehicleoccupancy'] == '8') & (df['meansoftransport'] == '1')] = 8
    df['occupancytransport'][(df['vehicleoccupancy'] == '9') & (df['meansoftransport'] == '1')] = 9
    df['occupancytransport'][(df['vehicleoccupancy'] == '10') & (df['meansoftransport'] == '1')] = 10
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '2')] = 11
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '3')] = 12
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '4')] = 13
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '5')] = 14
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '6')] = 15
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '7')] = 16
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '8')] = 17
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '9')] = 18
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '10')] = 19
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '11')] = 20
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '12')] = 21
    df['occupancytransport'][(df['vehicleoccupancy'] == '?') & (df['meansoftransport'] == '?')] = 22

    df['occupancytransport'].value_counts()

    df.drop(['vehicleoccupancy', 'meansoftransport'], axis = 1, inplace = True)

    df['schoolenrollment'] = df['schoolenrollment'].replace(['?'], 0)

    df['educationalattain'] = df['educationalattain'].replace(['?'], 14)

    df['workarrivaltime'] = df['workarrivaltime'].replace(['?'], 0)
    df['workarrivaltime'] = pd.to_numeric(df['workarrivaltime'])
    df['workarrivaltime'][(df['workarrivaltime'] > 0) & (df['workarrivaltime'] < 87)] = 1
    df['workarrivaltime'][(df['workarrivaltime'] >= 87) & (df['workarrivaltime'] < 94)] = 2
    df['workarrivaltime'][(df['workarrivaltime'] >= 94) & (df['workarrivaltime'] < 100)] = 3
    df['workarrivaltime'][(df['workarrivaltime'] >= 100) & (df['workarrivaltime'] < 106)] = 4
    df['workarrivaltime'][(df['workarrivaltime'] >= 106) & (df['workarrivaltime'] < 112)] = 5
    df['workarrivaltime'][(df['workarrivaltime'] >= 112) & (df['workarrivaltime'] < 118)] = 6
    df['workarrivaltime'][(df['workarrivaltime'] >= 118) & (df['workarrivaltime'] < 142)] = 7
    df['workarrivaltime'][(df['workarrivaltime'] >= 142) & (df['workarrivaltime'] < 202)] = 8
    df['workarrivaltime'][(df['workarrivaltime'] >= 202) & (df['workarrivaltime'] <= 285)] = 9

    df = df[df.age > 16]

    df['hoursworkperweek'] = df['hoursworkperweek'].replace(['?'], 0)

    df.drop('ancestry', axis = 1, inplace = True)

    df['degreefield'] = df['degreefield'].replace(['?'], 1000)

    df['industryworkedin'] = df['industryworkedin'].replace(['?'], 0)
    df['industryworkedin'] = pd.to_numeric(df['industryworkedin'])


    df['industryworkedin'][(df['industryworkedin'] >= 170) & (df['industryworkedin'] < 370)] = 1
    df['industryworkedin'][(df['industryworkedin'] >= 370) & (df['industryworkedin'] < 570)] = 2
    df['industryworkedin'][(df['industryworkedin'] >= 570) & (df['industryworkedin'] < 770)] = 3
    df['industryworkedin'][(df['industryworkedin'] >= 770) & (df['industryworkedin'] < 1070)] = 4
    df['industryworkedin'][(df['industryworkedin'] >= 1070) & (df['industryworkedin'] < 4070)] = 5
    df['industryworkedin'][(df['industryworkedin'] >= 4070) & (df['industryworkedin'] < 4670)] = 6
    df['industryworkedin'][(df['industryworkedin'] >= 4670) & (df['industryworkedin'] < 6070)] = 7
    df['industryworkedin'][(df['industryworkedin'] >= 6070) & (df['industryworkedin'] < 6470)] = 8
    df['industryworkedin'][(df['industryworkedin'] >= 6470) & (df['industryworkedin'] < 6870)] = 9
    df['industryworkedin'][(df['industryworkedin'] >= 6870) & (df['industryworkedin'] < 7270)] = 10
    df['industryworkedin'][(df['industryworkedin'] >= 7270) & (df['industryworkedin'] < 7860)] = 11
    df['industryworkedin'][(df['industryworkedin'] >= 7860) & (df['industryworkedin'] < 7970)] = 12
    df['industryworkedin'][(df['industryworkedin'] >= 7970) & (df['industryworkedin'] < 8370)] = 13
    df['industryworkedin'][(df['industryworkedin'] >= 8370) & (df['industryworkedin'] < 8560)] = 14
    df['industryworkedin'][(df['industryworkedin'] >= 8560) & (df['industryworkedin'] < 8770)] = 15
    df['industryworkedin'][(df['industryworkedin'] >= 8770) & (df['industryworkedin'] < 9370)] = 16
    df['industryworkedin'][(df['industryworkedin'] >= 9370) & (df['industryworkedin'] < 9670)] = 17
    df['industryworkedin'][(df['industryworkedin'] >= 9670) & (df['industryworkedin'] < 9870)] = 18
    df['industryworkedin'][(df['industryworkedin'] == 9920)] = 0
    
    df = pd.get_dummies(df, columns = ['workerclass', 'occupancytransport', 'schoolenrollment', 'educationalattain',
                                  'workarrivaltime', 'degreefield', 'industryworkedin', 'sex', 'marital'])

    return df


# In[5]:


# Clean train and test dataframes

# Train data
X_train = clean_dataframe(df_train)
y_train = X_train['wages']
X_train.drop('wages', axis=1, inplace=True)
X_train.drop('idnum', axis=1, inplace=True)

# Test data
X_test = clean_dataframe(X_test)
y_test_idnum = X_test['idnum']
X_test.drop('idnum', axis=1, inplace=True)


# In[6]:


# Match dummy variables in test and train

# Get missing columns in the training test
missing_cols = set(X_train.columns) - set(X_test.columns)
# Add a missing column in test set with default value equal to 0
for col in missing_cols:
    X_test[col] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]


# In[7]:


# Model 1 : DecisionTree Regressor : Finding parameters
# Also using RFE - Recursive feature elimination

# Model No. 1
# Decision Tree Regressor

tree_model = DecisionTreeRegressor(max_depth=1)
rfe_tree_model = RFE(estimator=tree_model, n_features_to_select=50, step=20)

# Validation - Get scores, no need to get predictions on the train data. Use above model to predict on test
scores = cross_validate(estimator=rfe_tree_model, X=X_train, y=y_train, cv=8,                         scoring='neg_mean_squared_error', return_estimator=False)
rmse_cv_tree = (-scores['test_score'].sum())**0.5
# print(scores)

# Fit model
rfe_tree_model.fit(X_train, y_train)

# Predict on the preprocessed test data
y_test = rfe_tree_model.predict(X_test)

# print(y_test)


# In[8]:


# Model 2: Random Forest Regressor

forest_model = RandomForestRegressor(max_depth=5)
rfe_forest_model = RFE(estimator=forest_model, n_features_to_select=50, step=20)

# Validation - Get scores, no need to get predictions on the train data. Use above model to predict on test
scores = cross_validate(estimator=rfe_forest_model, X=X_train, y=y_train, cv=8,                         scoring='neg_mean_squared_error', return_estimator=False)
rmse_cv_forest = ((-scores['test_score'].sum())**0.5)
# print(scores)

# Fit model
rfe_forest_model.fit(X_train, y_train)

# Predict on the preprocessed test data
y_test = rfe_forest_model.predict(X_test)

# print(y_test)


# In[10]:


predictions = pd.DataFrame(index=y_test_idnum, data=y_test)

# Creating output data and writing to csv

predictions.columns = ['Wages']
predictions.index.names = ['Id']

# Create csv file
predictions.to_csv('test_outputs.csv', sep=',')


# print('\n\nTree Error: ', rmse_cv_tree)
# print('Forest Error: ',rmse_cv_forest)
