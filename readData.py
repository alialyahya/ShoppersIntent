import csv
import numpy as np
import random
import pandas as pd
from setuptools.command.test import test

categorical = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

finalData = []
arrayKeys = {}
with open('rawData.csv', newline='') as csvFile:
    dataLine = csv.reader(csvFile)
    headers = next(dataLine)
    for i, label in enumerate(headers):
        arrayKeys[label] = i
    index = 0
    for row in dataLine:
        currentRow = [index] + [float(x) for x in row[:10]]
        index += 1
        currentRow.extend(row[10:])
        finalData.append(currentRow)

trueData = []
falseData = []

ratioTrue = 0
for i in finalData:
    if i[-1] == 'FALSE':
        falseData.append(i)
    elif i[-1] == 'TRUE':
        trueData.append(i)
        # ratioTrue+=1

# print('in all ratio of true/all: ',ratioTrue/len(finalData))

testData = random.sample(trueData, int(.3 * len(trueData)))
testData.extend(random.sample(falseData, int(.3 * len(falseData))))

trainData = []

for i in finalData:
    if i not in testData:
        trainData.append(i)

trainData = np.array(trainData)[:, 1:]
testData = np.array(testData)[:, 1:]

# ratioTrue = 0
# for i in testData:
#     if i[-1] == 'TRUE':
#         ratioTrue += 1

# print('in test ratio of true/all: ',ratioTrue/len(testData))


# ratioTrue = 0
# for i in trainData:
#     if i[-1] == 'TRUE':
#         ratioTrue += 1

# print('in train ratio of true/all: ',ratioTrue/len(trainData))
