import random
from readData import trainData


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
        self.smote()

    def upsample(self):
        self.upsampled = self.falseData[:]
        self.upsampled.extend(random.choices(self.trueData, k=len(self.falseData)))

    def downsample(self):
        self.downsampled = self.trueData[:]
        self.downsampled.extend(random.choices(self.falseData,k=len(self.trueData)))




# trueCount = 0
# falseCount = 0
# for i in test.downsampled:
#     if i[-1]=='TRUE':
#         trueCount+=1
#     elif i[-1]== 'FALSE':
#         falseCount+=1
#
# print(trueCount==falseCount,trueCount+falseCount==len(test.downsampled))