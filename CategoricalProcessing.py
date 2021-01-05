from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from readData import trainData
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)
categories = np.array([
    np.array(['Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep'],dtype='<U32'),
    np.array(['1', '2', '3', '4', '5', '6', '7', '8'],dtype='<U32'),
    np.array(['1', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U32'),
    np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9'],dtype='<U32'),
    np.array(['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '3', '4', '5', '6', '7', '8', '9'], dtype='<U32'),
    np.array(['New_Visitor', 'Other', 'Returning_Visitor'],dtype='<U32'),
    np.array(['FALSE', 'TRUE'],dtype='<U32')
])



# normal features are 0-9
# month is feature 10 for ordinal
# categorical features are 11-17

class processCategorical:

    def __init__(self, data):
        self.cleanAll = (data[:, :10].astype(np.float),data[:,-1])
        self.oneHotEncoder(data)

    def oneHotEncoder(self, data):
        enc = OneHotEncoder(categories=categories,handle_unknown='ignore')
        # enc = OneHotEncoder()
        subEncoded = enc.fit_transform(data[:, 10:-1])
        self.oneHot = data[:, :10]
        self.oneHot = np.append(self.oneHot, subEncoded.toarray(), axis=1)
        self.oneHotAll = (self.oneHot.astype(np.float), data[:, -1])


class smoteData:
    """preform this only on data sets with no upsampling/downsampling"""

    def __init__(self, dataAll):
        self.sm(dataAll)
        self.ada(dataAll)

    def sm(self, data):
        smt = SMOTE()
        X = data[0]
        y = data[1]
        Xnew, ynew = smt.fit_resample(X, y)
        self.smotedData = (Xnew,ynew)

    def ada(self, data):
        adasyn = ADASYN()
        X = data[0]
        y = data[1]
        Xnew, ynew = adasyn.fit_resample(X, y)
        self.adaData = (Xnew, ynew)



# enc = OneHotEncoder(categories=categories,handle_unknown='ignore')
# subEncoded = enc.fit_transform(trainData[:, 10:-1])
