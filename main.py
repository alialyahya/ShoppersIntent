from readData import testData
from dataHolder import dealtData
from sklearn.pipeline import Pipeline
from CategoricalProcessing import processCategorical
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)
#
# upOne = dealtData.allData['up onehot']
# X = upOne[0]
# y = upOne[1]
# dimReduc = PCA(n_components=7)
#
# reducedTrain = dimReduc.fit_transform(X)
#
# norm = Normalizer()
# normTrain = norm.fit_transform(reducedTrain)
#
# clf = RandomForestClassifier(ccp_alpha=0, class_weight='balanced',
#                              max_samples=0.8, n_estimators=200,
#                              n_jobs=-1)
#
# clf.fit(normTrain, y)
#
# # test data performance
# # ------------------------------------------------------------
#
# procTest = processCategorical(testData)
# testCat = procTest.oneHotAll[0]
#
# redTest = dimReduc.transform(testCat)
# finalTest = norm.transform(redTest)
#
# yTestTrue = procTest.oneHotAll[1]
#
# yPred = clf.predict(finalTest)
#
# print('please please please: ', f1_score(yTestTrue, yPred, pos_label='TRUE'))

procTest = processCategorical(testData)
testOrder = [procTest.cleanAll, procTest.cleanAll, procTest.oneHotAll, procTest.oneHotAll, procTest.cleanAll,
             procTest.oneHotAll]

dataOrder = ['raw clean', 'smote clean', 'raw onehot', "smote onehot", 'up clean', "up onehot"]

noDimReducPre = [ColumnTransformer(transformers=[('num', MinMaxScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))]),
                 ColumnTransformer(transformers=[('num', StandardScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))]),
                 ColumnTransformer(transformers=[('num', MinMaxScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))]),
                 ColumnTransformer(transformers=[('num', StandardScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))]),
                 ColumnTransformer(transformers=[('num', MinMaxScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))]),
                 ColumnTransformer(transformers=[('num', MinMaxScaler(),
                                                  slice(None, 10, None)),
                                                 ('cat', 'passthrough',
                                                  slice(10, None, None))])
                 ]

noDimReducClassifier = [RandomForestClassifier(ccp_alpha=0.01, class_weight='balanced',
                                               max_samples=0.8, n_estimators=150,
                                               n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=150, n_jobs=-1, class_weight = 'balanced'),
                        RandomForestClassifier(ccp_alpha=0.001,
                                               class_weight='balanced',
                                               max_samples=0.8, n_estimators=200,
                                               n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=150, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, class_weight='balanced',
                                               max_samples=0.8, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, class_weight='balanced',
                                               max_samples=0.8, n_estimators=150,
                                               n_jobs=-1)
                        ]

print("No Dimensionality Reduction:")
print('============================================================')
for i, label in enumerate(dataOrder):
    print(label)
    print('Best Classifier for no Dim Reduc and', label, 'dataset.')
    X = dealtData.allData[label][0]
    y = dealtData.allData[label][1]

    prep = noDimReducPre[i]
    preppedX = prep.fit_transform(X)

    clf = noDimReducClassifier[i]
    clf.fit(preppedX, y)

    Xtest = testOrder[i][0]
    yTest = testOrder[i][1]

    preppedTest = prep.transform(Xtest)
    yPred = clf.predict(preppedTest)

    print('F1 score:', f1_score(yTest, yPred, pos_label='TRUE'))
    print("classifier", clf)
    print('---------------------------------------------------------------')

# with dim reduc
# ---------------------------------------------------------------

procTest = processCategorical(testData)
testOrder = [procTest.cleanAll, procTest.cleanAll, procTest.oneHotAll, procTest.oneHotAll, procTest.cleanAll,
             procTest.oneHotAll]

dataOrder = ['raw clean', 'smote clean', 'raw onehot', "smote onehot", 'up clean', "up onehot"]

dimReducer = [PCA(n_components=6), PCA(n_components=8), PCA(n_components=37), PCA(n_components=18), PCA(n_components=6),
              PCA(n_components=7)]

DimReducePre = [StandardScaler(), StandardScaler(), MinMaxScaler(), StandardScaler(), Normalizer(), Normalizer()]

DimReducClassifier = [RandomForestClassifier(ccp_alpha=0.001,
                                               class_weight='balanced',
                                               max_samples=0.8, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=50, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0.001,
                                               class_weight='balanced',
                                               max_samples=0.8, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=200, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=50, n_jobs=-1),
                        RandomForestClassifier(ccp_alpha=0, max_samples=0.8,
                                               n_estimators=150, n_jobs=-1)
                        ]

print("With Dimensionality Reduction:")
print('============================================================')
for i, label in enumerate(dataOrder):
    print(label)
    print('Best Classifier for Dim Reduc and', label, 'dataset.')
    Xpre = dealtData.allData[label][0]
    y = dealtData.allData[label][1]

    dimmer = dimReducer[i]
    X= dimmer.fit_transform(Xpre)

    prep = DimReducePre[i]
    preppedX = prep.fit_transform(X)

    clf = DimReducClassifier[i]
    clf.fit(preppedX, y)

    XpreTest = testOrder[i][0]
    yTest = testOrder[i][1]

    Xtest = dimmer.transform(XpreTest)
    preppedTest = prep.transform(Xtest)
    yPred = clf.predict(preppedTest)
    print("classifier", clf)
    print('F1 score:', f1_score(yTest, yPred, pos_label='TRUE'))
    print('---------------------------------------------------------------')

clf = LinearSVC( C= .0001, class_weight = 'balanced')
X= dealtData.allData['raw onehot'][0]
y= dealtData.allData['raw onehot'][1]
clf.fit(X,y)

Xtest = procTest.oneHotAll[0]
ytest = procTest.oneHotAll[1]
ypred = clf.predict(Xtest)
print('--------------------------------------------------------------------')
print("Test data on best Baseline system: ")
print("F1 score", f1_score(ytest,ypred,pos_label="TRUE"))