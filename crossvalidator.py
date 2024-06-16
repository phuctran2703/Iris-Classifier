import numpy as np
from tools import Tools
from classificFunction import ClassificFunction

class CrossValidator:
    def __init__(self, data, target, classes, k=5):
        self.data = data
        self.target = target
        self.classes = classes
        self.k = k
        self.foldSize = len(data) // k
        self.indices = np.arange(len(data))
        np.random.shuffle(self.indices)

    def splitData(self, fold):
        testIdx = self.indices[fold * self.foldSize:(fold + 1) * self.foldSize]
        trainIdx = np.concatenate((self.indices[:fold * self.foldSize], self.indices[(fold + 1) * self.foldSize:]))
        return self.data[trainIdx], self.target[trainIdx], self.data[testIdx], self.target[testIdx]

    def updateConfusionMatrix(self, confusionMatrix, actualLabels, predictedLabels):
        for actualLabel, predLabel in zip(actualLabels, predictedLabels):
            confusionMatrix[actualLabel, predLabel] += 1

    def crossValidate(self, classifyFunc):
        accuracies = []
        confusionMatrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)

        for fold in range(self.k):
            dataTrain, targetTrain, dataTest, targetTest = self.splitData(fold)
            predictedClasses = classifyFunc(dataTrain, targetTrain, dataTest)
            correct = np.sum(predictedClasses == targetTest)
            total = len(targetTest)
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy)
            self.updateConfusionMatrix(confusionMatrix, targetTest, predictedClasses)

        return np.mean(accuracies), confusionMatrix

    def classifyOneVersusRest(self, dataTrain, targetTrain, dataTest):
        class1DiscFunc = ClassificFunction.oneVersusRest(dataTrain, targetTrain, self.classes[0])
        class2DiscFunc = ClassificFunction.oneVersusRest(dataTrain, targetTrain, self.classes[1])

        predictionlist = []
        for x in dataTest:
            predictions1 = class1DiscFunc(x)
            predictions2 = class2DiscFunc(x)
            
            if(predictions1[0] > predictions1[1] and predictions2[0] < predictions2[1]): predictionlist.append(0)
            elif(predictions1[0] < predictions1[1] and predictions2[0] > predictions2[1]): predictionlist.append(1)
            elif(predictions1[0] < predictions1[1] and predictions2[0] < predictions2[1]): predictionlist.append(2)
            else: predictionlist.append(0)

        return np.array(predictionlist)

    def classifyOneVersusOne(self, dataTrain, targetTrain, dataTest):
        class1vsclass2DiscFunc = ClassificFunction.oneVersusOne(dataTrain, targetTrain, self.classes[0], self.classes[1])
        class1vsclass3DiscFunc = ClassificFunction.oneVersusOne(dataTrain, targetTrain, self.classes[0], self.classes[2])
        class2vsclass3DiscFunc = ClassificFunction.oneVersusOne(dataTrain, targetTrain, self.classes[1], self.classes[2])

        predictionlist = []
        for x in dataTest:
            predictions = []
            predictions1 = class1vsclass2DiscFunc(x)
            predictions2 = class1vsclass3DiscFunc(x)
            predictions3 = class2vsclass3DiscFunc(x)
            
            if(predictions1[0] > predictions1[1]): predictions.append(0)
            else: predictions.append(1)
            if(predictions2[0] > predictions2[1]): predictions.append(0)
            else: predictions.append(2)
            if(predictions3[0] > predictions3[1]): predictions.append(1)
            else: predictions.append(2)
            
            count0 = np.count_nonzero(np.array(predictions) == 0)
            count1 = np.count_nonzero(np.array(predictions) == 1)
            count2 = np.count_nonzero(np.array(predictions) == 2)

            if count0 > count1 and count0 > count2: predictionlist.append(0)
            elif count1 > count0 and count1 > count2: predictionlist.append(1)
            elif count2 > count0 and count2 > count1: predictionlist.append(2)
            else:
                predictionlist.append(0)

        return np.array(predictionlist)

    def classifyThreeClass(self, dataTrain, targetTrain, dataTest):
        discriminantFunction = ClassificFunction.threeClassDiscriminate(dataTrain, targetTrain, *self.classes)
        predictions = np.array([discriminantFunction(x) for x in dataTest])
        return np.argmax(predictions, axis=1)