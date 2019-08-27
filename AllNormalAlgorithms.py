import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#With All Attributes code
dataset = pd.read_csv('unbc-trainingset1.csv')
X_train = dataset.iloc[:, :-1].values 
y_train = dataset.iloc[:, 41].values  

dataset1 = pd.read_csv('unbc-testingset1.csv')
X_test = dataset1.iloc[:, :-1].values 
y_test = dataset1.iloc[:, 41].values 


#Reduced Attributes code 
dataset = pd.read_csv('unbc-trainingset1.csv')
X_train = dataset.iloc[:, [1,2,18,29,31,32,33,34,38,39]].values 
y_train = dataset.iloc[:, 41].values 

dataset1 = pd.read_csv('unbc-testingset1.csv')
X_test = dataset1.iloc[:, [1,2,18,29,31,32,33,34,38,39]].values 
y_test = dataset1.iloc[:, 41].values


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test) 



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)



from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#by default metric is minikowski, but hwen you put p = 2, it becomes eculedian Distance

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

