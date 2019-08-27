import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#With All Attributes code
dataset = pd.read_csv('unbc-trainingset1.csv')
x_train = dataset.iloc[:, :-1].values 
y_train = dataset.iloc[:, 41].values  

dataset1 = pd.read_csv('unbc-testingset1.csv')
x_test = dataset1.iloc[:, :-1].values 
y_test = dataset1.iloc[:, 41].values 

#Reduced Attributes code 
dataset = pd.read_csv('unbc-trainingset1.csv')
x_train = dataset.iloc[:, [1,2,5,6,7,8,25,32,33,34,39]].values 
y_train = dataset.iloc[:, 41].values 

dataset1 = pd.read_csv('unbc-testingset1.csv')
x_test = dataset1.iloc[:, [1,2,5,6,7,8,25,32,33,34,39]].values 
y_test = dataset1.iloc[:, 41].values








import xgboost as xgb
classifier=xgb.XGBClassifier(random_state=1,learning_rate=0.01)





from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(random_state=10)



from sklearn.ensemble import BaggingClassifier
from sklearn import tree
classifier = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))



from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(learning_rate=0.01,random_state=1)


from time import time
t0 = time()
classifier.fit(x_train, y_train)
print("training time:", round(time()-t0, 3), "s")
t1 = time()
y_pred = classifier.predict(x_test)
print("Predict time:", round(time()-t1, 3), "s")
 
#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)


#OverALL Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)




#Individual Accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# Other way to get overall accuracy from confusion matrix
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

print(accuracy(confusion_matrix))











