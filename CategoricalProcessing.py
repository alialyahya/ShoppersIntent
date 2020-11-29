from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


# normal features are 0-9
# month is feature 10 for ordinal
# categorical features are 11-17

class processCategorical:

    def __init__(self, data):
        self.clean = data[:, :10]
        self.oneHotEncoder(data)
        self.ordinalEncoder(data)

    def oneHotEncoder(self, data):
        enc = OneHotEncoder()
        subEncoded = enc.fit_transform(data[:, 10:])

        self.oneHot = data[:, :10]
        self.oneHot = np.append(self.oneHot, subEncoded, axis=1)

    def ordinalEncoder(self, data):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                  'Dec']

        enc = OneHotEncoder()
        subEncoded = enc.fit_transform(data[:, 11:])

        ordEnc = OrdinalEncoder([months])
        ordinalData = ordEnc.fit_transform(data[:, 10])

        self.ordinal = data[:, :10]
        self.ordinal = np.append(self.oneHot, ordinalData, axis=1)
        self.ordinal = np.append(self.oneHot, subEncoded, axis=1)


class smoteData:

    def __init__(self, data):
        pass
