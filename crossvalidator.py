import numpy as np
from model.fisher_model import FisherModel
from model.generative_model import GenerativeModel
from model.one_versus_one_model import OneVersusOneModel
from model.one_versus_rest_model import OneVersusRestModel
from model.multiple_classes_model import MulticlassModel

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
        if (classifyFunc.__name__ == "classifyFisher"):
            self.data, self.target = FisherModel(self.data, self.target, self.classes).projectToTwoDimensions()
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
        model = OneVersusRestModel(dataTrain, targetTrain, self.classes)
        model.trainModel()
        predictions = np.array([model.predict(x) for x in dataTest])
        return predictions

    def classifyOneVersusOne(self, dataTrain, targetTrain, dataTest):
        model = OneVersusOneModel(dataTrain, targetTrain, self.classes)
        model.trainModel()
        predictions = np.array([model.predict(x) for x in dataTest])
        return predictions

    def classifyThreeClass(self, dataTrain, targetTrain, dataTest):
        model = MulticlassModel(dataTrain, targetTrain, self.classes)
        model.trainModel()
        predictions = np.array([model.predict(x) for x in dataTest])
        return predictions
    
    def classifyFisher(self, dataTrain, targetTrain, dataTest):
        model = FisherModel(dataTrain, targetTrain, self.classes)
        model.trainModel()
        predictions = np.array([model.predict(x) for x in dataTest])
        return predictions
    
    def classifyGeneration(self, dataTrain, targetTrain, dataTest):
        model = GenerativeModel(dataTrain, targetTrain, self.classes)
        model.trainModel()
        predictions = np.array([model.predict(x) for x in dataTest])
        return predictions