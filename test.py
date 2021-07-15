import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
import os

print('Load data...',end='')
filepath = os.path.join('.','test_set.csv')
test_set = np.genfromtxt(filepath, delimiter=',', skip_header=1)
print('DONE')

# features
X = test_set[:, :-1]

# labels
y = test_set[:, -1]

# Preprocessing
preprocess = joblib.load('preprocess.pkl')

print('Preprocessing...', end='')
X = preprocess.transform(X)
print('DONE')

# Load SVM
clf = joblib.load('model.pkl')

y_predict = clf.predict(X)
labels = sorted(list(set(np.unique(y))))
print("\nConfusion matrix:")
print(confusion_matrix(y, y_predict, labels=labels))

print("\nClassification report:")
print(classification_report(y, y_predict))
