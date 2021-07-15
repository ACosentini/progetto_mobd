import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import SGDClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.ensemble import AdaBoostClassifier

# Set seed
np.random.seed = 1

print('Load data...',end='')
filepath = os.path.join('.','training_set.csv')
traning_set = np.genfromtxt(filepath, delimiter=',', skip_header=1)
print('DONE')

print(f"Training set shape: {traning_set.shape}")

test = np.concatenate([traning_set[np.where(traning_set[:,-1] == 0)][:598],
                       traning_set[np.where(traning_set[:,-1] == 1)][:311],
                       traning_set[np.where(traning_set[:,-1] == 2)][:492],
                       traning_set[np.where(traning_set[:,-1] == 3)][:599]])

train = np.concatenate([traning_set[np.where(traning_set[:,-1] == 0)][598:],
                        traning_set[np.where(traning_set[:,-1] == 1)][311:],
                        traning_set[np.where(traning_set[:,-1] == 2)][492:],
                        traning_set[np.where(traning_set[:,-1] == 3)][599:]])


# Preprocessing
preprocess = Pipeline([('imputer', KNNImputer(n_neighbors=50)),
                       ('scaler', RobustScaler())])

# features
X_train = train[:, :-1]
X_test = test[:, :-1]

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

if not os.path.exists("preprocess.pkl"):
    joblib.dump(preprocess, "preprocess.pkl")
else:
    print("Cannot save trained preprocess model to {0}.".format("preprocess.pkl"))

# labels
y_train = train[:,-1]
y_test = test[:,-1]




def train_svm_classifer(X_train, y_train, X_test, y_test, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """

    param = [
        #{
        #    "kernel": ["poly"],
        #    "C": [0.01, 0.1, 1, 10, 100, 1000],
        #    "degree": [3,5,7,9]
        #},
        #{
        #    "kernel": ["sigmoid"],
        #    "C": [0.01,0.1,1, 10, 100, 1000]
        #},
        {
            "kernel": ["rbf"],
            "C": [2.5, 3, 5],#,2,2.5,3,5,7],#, 10, 100,1000],
            "gamma": [0.1],#,0.1,0.11,0.15,0.2],# ,1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    # request probability estimation
    svm = SVC(probability=True)

    # 4-fold cross validation, use n thread as each fold and each parameter set can be train in parallel
    clf = GridSearchCV(svm, param,
            cv=4, n_jobs=-1, verbose=3)

    clf.fit(X_train, y_train)

    if not os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)
    y_predict=clf.predict(X_train)
    print("\nClassification report (train):")
    print(classification_report(y_train, y_predict))

    y_predict=clf.predict(X_test)

    labels=sorted(list(set(np.unique(y_predict))))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report (test):")
    print(classification_report(y_test, y_predict))


train_svm_classifer(X_train, y_train, X_test, y_test, './model.pkl')
