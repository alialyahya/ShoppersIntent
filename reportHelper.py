from dataHolder import dealtData
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,f1_score
import warnings
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)


# thank you to user serlouk for help in saving gridsearchcv objects
# source: https://stackoverflow.com/questions/51424312/how-to-save-gridsearchcv-object

dataRep = [ "up onehot_withDim.pkl","up onehot_noDim.pkl","up clean_withDim.pkl","up clean_noDim.pkl",
            "smote onehot_withDim.pkl","smote onehot_noDim.pkl","smote clean_withDim.pkl","smote clean_noDim.pkl",
            "raw onehot_withDim.pkl","raw onehot_noDim.pkl","raw clean_withDim.pkl","raw clean_noDim.pkl"
           ]

bestLogit = {}
for drep in dataRep:
    test = joblib.load(drep)
    currentValue = len(test.cv_results_['rank_test_score'])
    for i, value in enumerate(test.cv_results_['rank_test_score']):
        if value<currentValue and test.cv_results_['params'][i]['classify'].__class__.__name__ == 'LogisticRegression':
            currentValue = value
            best = (test.cv_results_['params'][i],test.cv_results_['mean_test_score'][i],test.cv_results_['std_test_score'][i])
    bestLogit[drep]=best

for i in bestLogit:
    print("For",i)
    print("mean",bestLogit[i][1])
    print("std", bestLogit[i][2])
    print("params", bestLogit[i][0])
    print('---------------------------------------------------------------------')