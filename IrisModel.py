import numpy as numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def irisFlowerModel():
    #import data
    irisDf = pd.read_csv("C:\\Users\\hcheena\\OneDrive - KPMG\\Documents\\Iris Flower\\IRIS.csv")
    #encode the output
    labels = irisDf['species']
    lb = LabelBinarizer()
    oneHotEncodedLabels = lb.fit_transform(labels)
    oneHotEncodedLabels = pd.DataFrame(oneHotEncodedLabels, columns=lb.classes_)

    #form the features dataframe
    #drop the species field
    features = irisDf.drop("species", axis=1)

    #train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(features, oneHotEncodedLabels)

    return clf,lb.classes_

