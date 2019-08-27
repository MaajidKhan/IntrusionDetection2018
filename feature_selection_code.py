# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:27:00 2019

@author: maajdkhan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('unbc-trainingset1.csv')
dataset = dataset.select_dtypes(include=[np.number]).interpolate().dropna()
X_train = dataset.iloc[:, :-1].values 
y_train = dataset.iloc[:, 41].values  
 
dataset1 = pd.read_csv('unbc-testingset1.csv')
dataset1 = dataset.select_dtypes(include=[np.number]).interpolate().dropna()
X_test = dataset1.iloc[:, :-1].values 
y_test = dataset1.iloc[:, 41].values 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)



model = classifier.fit(X_train, y_train)

#code4
import pandas as pd
%matplotlib inline
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(dataset.columns, classifier.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance')