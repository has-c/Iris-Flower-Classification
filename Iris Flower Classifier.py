from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd

#import iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, 
                  names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
#note: target names are the labels and the features are the actual data

#split dataset into a training and test dataframe
n = len(iris)
is_train = np.random.rand(n) < 0.7
train = iris[is_train].reset_index(drop=True)
test = iris[~is_train].reset_index(drop=True)

#split species from rest of data
train_labels = train['species'].copy()
train = train.drop('species', axis=1)
test_labels = test['species'].copy()
test = test.drop('species', axis=1)

#create decision tree classifier
clf = tree.DecisionTreeClassifier()
#clf.fit(features, labels) - general form of the fit command
clf.fit(train, train_labels)


#use testing data to test the accuracy of the model
print("Test Outputs")
#output we want to see
print(test_labels)

print() #blank line 

output = clf.predict(test) #prediction using decision tree
print(output)
print('Prediction Outputs')



