#import modules
from sklearn import tree
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix 

#evaluate accuracy of the model 
#calculate RMSE (root mean squared error) to evaluate accuracy 
def rmse(y, yhat):
    return np.sum((y - yhat)**2)**0.5

#import iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, 
                  names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
#note: target names are the labels and the features are the actual data

#split into features and labels
data = iris.drop('species', axis = 1)
labels = iris['species']
#split into training and testing datasets
train, test, train_labels, test_labels = train_test_split(data, labels, test_size = 0.20) 

#predict species of flower depending on the sepal length, sepal width, petal length & petal width
#create decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train, train_labels)

#predictions 
output = clf.predict(test) #prediction using decision tree

print(confusion_matrix(test_labels, output))
print(classification_report(test_labels, output))




