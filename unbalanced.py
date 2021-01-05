import random
from readData import trainData
import numpy as np
# need to shuffle data

# Thank you to Jason Brownlee for providing resources on how to use SMOTE to balance data
# source: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
class BalancedData:

    def __init__(self, data):
        self.data = data
        falseData = []
        trueData = []
        for i in data:
            if i[-1] == 'FALSE':
                falseData.append(i)
            elif i[-1] == 'TRUE':
                trueData.append(i)
        self.falseData = falseData
        self.trueData = trueData

        self.upsample()
        self.downsample()

    def upsample(self):
        self.upsampled = self.falseData[:]
        self.upsampled.extend(random.choices(self.trueData, k=len(self.falseData)))
        self.upsampled = np.array(self.upsampled)

    def downsample(self):
        self.downsampled = self.trueData[:]
        self.downsampled.extend(random.choices(self.falseData,k=len(self.trueData)))
        self.downsampled = np.array(self.downsampled)



# trueCount = 0
# falseCount = 0
# for i in test.downsampled:
#     if i[-1]=='TRUE':
#         trueCount+=1
#     elif i[-1]== 'FALSE':
#         falseCount+=1
#
# print(trueCount==falseCount,trueCount+falseCount==len(test.downsampled))