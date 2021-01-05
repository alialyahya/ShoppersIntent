from dataHolder import dealtData
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore',category=ConvergenceWarning)

# also perform for raw onehot
useThis = dealtData.allData['raw onehot']


# Thank you to user rossxsy for the tutorial on using gridsearchcv
# Source: https://www.kaggle.com/rossxsy/testing-multiple-classifiers

models = ['Perceptron', 'DummyStrat', 'DummyPop', 'GaussianNB','Linear SVC']

clfs = [Perceptron(max_iter=20000), DummyClassifier(strategy='stratified'), DummyClassifier(strategy='most_frequent'),
        GaussianNB(),  LinearSVC(max_iter=20000)]

params = {models[0]: {'penalty': ['l2', 'l1'], 'class_weight': [None, 'balanced'],
                      'alpha': list(np.append(np.array([0.0]), np.logspace(-4, 2, 7)))},
          models[1]: {},
          models[2]: {},
          models[3]: {},
          models[4]: {'penalty': ['l2', 'l1'],'class_weight': [None,'balanced'], 'C': list(np.logspace(-4, 2, 7))}
          }

# for name, estimator in zip(models, clfs):
#     print(name)
#     clf = GridSearchCV(estimator, params[name],
#                        scoring= make_scorer(f1_score,pos_label='TRUE'))
#     X, y = useThis
#     clf.fit(X,y)
#     print("best params: " + str(clf.best_params_))
#     print("best scores: " + str(clf.best_score_))
#     print('--------------------------------------------')
#

clf = GridSearchCV(clfs[4],params[models[4]],scoring= make_scorer(f1_score,pos_label='TRUE'))



X, y = useThis
clf.fit(X,y)

indeces = []
index = 0
for i, value in enumerate(clf.cv_results_['rank_test_score']):
    if value<=5:
        print('rank:', value)
        print('mean test score:', clf.cv_results_['mean_test_score'][i])
        print('std test score:', clf.cv_results_['std_test_score'][i])
        print('params:', clf.cv_results_['params'][i])

