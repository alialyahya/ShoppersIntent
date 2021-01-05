import csv
import numpy as np
categorical = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

testData = []
arrayKeys = {}
with open('.\\data\\test_data.csv', newline='') as csvFile:
    dataLine = csv.reader(csvFile)
    headers = next(dataLine)
    for i, label in enumerate(headers):
        arrayKeys[label] = i
    for row in dataLine:
        currentRow = [float(x) for x in row[:10]]
        currentRow.extend(row[10:])
        testData.append(currentRow)

testData = np.array(testData)

trainData = []
arrayKeys = {}

with open('.\\data\\train_data.csv', newline='') as csvFile:
    dataLine = csv.reader(csvFile)
    headers = next(dataLine)
    for i, label in enumerate(headers):
        arrayKeys[label] = i
    for row in dataLine:
        currentRow = [float(x) for x in row[:10]]
        currentRow.extend(row[10:])
        trainData.append(currentRow)

trainData = np.array(trainData)
