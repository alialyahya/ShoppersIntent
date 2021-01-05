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

# ColumnTransformer([('all', 'passthrough', slice(0, None))]),
# ColumnTransformer([('all', Binarizer(), slice(0, None))])

noFeat = [
          ColumnTransformer([('num', MinMaxScaler(), slice(10)), ('cat', 'passthrough', slice(10, None))]),
          ColumnTransformer([('num', StandardScaler(), slice(10)), ('cat', 'passthrough', slice(10, None))])
          ]

pipeNoFeat = Pipeline([
    ('preprocess', 'passthrough'),
    ('classify', 'passthrough')
])

param_gridNoFeat = [
    {
        'preprocess': noFeat,
        'classify': [LogisticRegression(max_iter=1000, n_jobs=-1)],
        'classify__penalty':['l1','l2'],
        'classify__C': list(np.logspace(-3, 2, 6)),
        'classify__class_weight': [None, "balanced"]
    },
    {
        'preprocess': noFeat,
        'classify': [SVC(max_iter=1000)],
        'classify__kernel': ['poly', 'rbf', 'sigmoid'],
        'classify__degree': [1, 2, 3],
        'classify__gamma': list(np.logspace(-3, 2, 6)),
        'classify__class_weight': [None, 'balanced']
    },
    {
        'preprocess': noFeat,
        'classify': [KNeighborsClassifier(n_jobs=-1)],
        'classify__n_neighbors': [50, 100, 150, 200],
    },
    {
        'preprocess': noFeat,
        'classify': [RandomForestClassifier(n_jobs=-1, max_samples=.8)],
        'classify__n_estimators': [50, 100, 150, 200],
        'classify__class_weight': ['balanced', None],
        'classify__ccp_alpha': [0] + list(np.logspace(-3, 2, 6))
    },
    {
        'preprocess': noFeat,
        'classify': [AdaBoostClassifier()],
        'classify__n_estimators': [50, 100, 150, 200]
    }
]

# np preprocessing
print('No Preprocessing:')
print('========================================================')
for i in dealtData.allData:
    print('Data type:', i)
    grid = GridSearchCV(pipeNoFeat, param_grid=param_gridNoFeat,
                        scoring=make_scorer(f1_score,pos_label='TRUE'))
    X, y = dealtData.allData[i]
    grid.fit(X, y)
    print('Best Score:', grid.best_score_)
    print('Best Estimator:', grid.best_estimator_)
    print("------------------------------------------------------------")
    joblib.dump(grid, i + '_noDim' + '.pkl')

# With Dimensionality Reduction
# --------------------------------------------------------------------------------------------------------------------------------------------


clean = ['raw clean', 'smote clean', 'up clean']

featureExtract = [
    PCA()
]

preprocess = [
    MinMaxScaler(),
    Normalizer(),
    StandardScaler(),
]

pipeNoFeat = Pipeline([
    ('dim_reduc', 'passthrough'),
    ('preprocess', 'passthrough'),
    ('classify', 'passthrough')
])

param_gridUnclean = [
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [int(.1 * 74), int(.25 * 74), int(.5 * 74)],
        'preprocess': preprocess,
        'classify': [LogisticRegression(max_iter=1000, n_jobs=-1)],
        'classify__penalty':['l1','l2'],
        'classify__C': list(np.logspace(-4, 2, 6)),
        'classify__class_weight': [None, "balanced"]
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [int(.1 * 74), int(.25 * 74), int(.5 * 74)],
        'preprocess': preprocess,
        'classify': [SVC(max_iter=1000)],
        'classify__kernel': ['poly', 'rbf', 'sigmoid'],
        'classify__degree': [1, 2, 3],
        'classify__gamma': list(np.logspace(-3, 2, 6)),
        'classify__class_weight': [None, 'balanced']
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [int(.1 * 74), int(.25 * 74), int(.5 * 74)],
        'preprocess': preprocess,
        'classify': [KNeighborsClassifier(n_jobs=-1)],
        'classify__n_neighbors': [50, 100, 150, 200],
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [int(.1 * 74), int(.25 * 74), int(.5 * 74)],
        'preprocess': preprocess,
        'classify': [RandomForestClassifier(n_jobs=-1, max_samples=.8)],
        'classify__n_estimators': [50, 100, 150, 200],
        'classify__class_weight': ['balanced', None],
        'classify__ccp_alpha': [0] + list(np.logspace(-3, 2, 6))
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [int(.1 * 74), int(.25 * 74), int(.5 * 74)],
        'preprocess': preprocess,
        'classify': [AdaBoostClassifier()],
        'classify__n_estimators': [50, 100, 150, 200]
    }
]

param_gridClean = [
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [2, 4, 6, 8],
        'preprocess': preprocess,
        'classify': [LogisticRegression(max_iter=1000, n_jobs=-1)],
        'classify__penalty':['l1','l2'],
        'classify__C': list(np.logspace(-4, 2, 6)),
        'classify__class_weight': [None, "balanced"]
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [2, 4, 6, 8],
        'preprocess': preprocess,
        'classify': [SVC(max_iter=1000)],
        'classify__kernel': ['poly', 'rbf', 'sigmoid'],
        'classify__degree': [1, 2, 3],
        'classify__gamma': list(np.logspace(-3, 2, 6)),
        'classify__class_weight': [None, 'balanced']
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [2, 4, 6, 8],
        'preprocess': preprocess,
        'classify': [KNeighborsClassifier(n_jobs=-1)],
        'classify__n_neighbors': [50, 100, 150, 200],
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [2, 4, 6, 8],
        'preprocess': preprocess,
        'classify': [RandomForestClassifier(n_jobs=-1, max_samples=.8)],
        'classify__n_estimators': [50, 100, 150, 200],
        'classify__class_weight': ['balanced', None],
        'classify__ccp_alpha': [0] + list(np.logspace(-3, 2, 6))
    },
    {
        'dim_reduc': featureExtract,
        'dim_reduc__n_components': [2, 4, 6, 8],
        'preprocess': preprocess,
        'classify': [AdaBoostClassifier()],
        'classify__n_estimators': [50, 100, 150, 200]
    }
]

print('With Dimensionality Reduction:')
print('========================================================')
for i in dealtData.allData:
    print('Data type:', i)
    X, y = dealtData.allData[i]
    if i in clean:
        grid = GridSearchCV(pipeNoFeat, param_grid=param_gridClean,
                            scoring=make_scorer(f1_score,pos_label='TRUE'))
        grid.fit(X, y)
    else:
        grid = GridSearchCV(pipeNoFeat, param_grid=param_gridUnclean,
                            scoring=make_scorer(f1_score,pos_label='TRUE'))
        grid.fit(X, y)

    print('Best Score:', grid.best_score_)
    print('Best Estimator:', grid.best_estimator_)
    print("------------------------------------------------------------")
    joblib.dump(grid, i + '_withDim' + '.pkl')

# pipe = Pipeline([
#     ('reduce_dim','passthrough')
#     ('preprocess', 'passthough'),
#     ('classify', 'passthrough')
# ])
