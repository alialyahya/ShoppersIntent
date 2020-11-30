from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from readData import trainData
from scipy.sparse import csr_matrix


# normal features are 0-9
# month is feature 10 for ordinal
# categorical features are 11-17

class processCategorical:

    def __init__(self, data):
        self.cleanAll = (data[:, :10].astype(np.float),data[:,-1])
        self.oneHotEncoder(data)

    def oneHotEncoder(self, data):
        enc = OneHotEncoder()
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





testCat = processCategorical(trainData)
smoteTest = smoteData(testCat.oneHotAll)
