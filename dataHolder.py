from readData import trainData,testData
from unbalanced import BalancedData
from CategoricalProcessing import processCategorical, smoteData

class dataHolder:
    """class to hold all types of data"""

    def __init__(self,balData):
        '''holds 8 versions of the data as follows: 1. no resampling with removed categ/
        2. smoted data with removed categ /  3. no resampling with one hot/
        4. smoted data with one hot/  5. upsampled data with removed categ/
        6. upsampled data with one hot/ 7. downsampled data with no categ/ 8. downsampled data with onehot'''
        self.allData={}
        rawProcess = processCategorical(balData.data)

        self.allData['raw clean'] = rawProcess.cleanAll
        tmpSmoteRaw = smoteData(self.allData['raw clean'])
        self.allData['smote clean'] = tmpSmoteRaw.smotedData

        self.allData['raw onehot'] = rawProcess.oneHotAll
        tmpSmoteOne = smoteData(self.allData['raw onehot'])
        self.allData['smote onehot'] = tmpSmoteOne.smotedData


        upProcess = processCategorical(balData.upsampled)
        self.allData['up clean'] = upProcess.cleanAll
        self.allData['up onehot'] = upProcess.oneHotAll

        # downProcess = processCategorical(balData.downsampled)
        # self.allData['down clean'] = downProcess.cleanAll
        # self.allData['down onehot'] = downProcess.oneHotAll






balanceData = BalancedData(trainData) # attributes: .data, .upsamples, downsampled

dealtData = dataHolder(balanceData)

