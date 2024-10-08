import numpy as np
from model.model import Model
from discriminant_function import ClassificFunction

class FisherModel(Model):
    def __init__(self, dataMatrix, target, classes):
        super().__init__(dataMatrix, target, classes)

    def _calculateMean(self):
        meanEachClassList = [np.mean(self.dataMatrix[self.target == cls], axis=0) for cls in self.classes]
        meanTotal = np.mean(self.dataMatrix, axis=0)
        return meanEachClassList, meanTotal

    def _calculateWithinClassCovarianceMatrix(self, meanEachClassList):
        covMatrixList = []
        for i, meanVec in enumerate(meanEachClassList):
            covMatrix = sum(np.dot((x - meanVec).reshape(-1, 1), (x - meanVec).reshape(1, -1)) 
                            for x in self.dataMatrix[self.target == self.classes[i]])
            covMatrixList.append(covMatrix)
        
        return sum(covMatrixList)

    def _calculateBetweenClassCovarianceMatrix(self, meanEachClassList, meanTotal):
        additionMatrix = sum(len(self.dataMatrix[self.target == cls]) * np.dot((meanVec - meanTotal).reshape(-1, 1), 
                                                                         (meanVec - meanTotal).reshape(1, -1)) 
                             for meanVec, cls in zip(meanEachClassList, self.classes))
        return additionMatrix

    def findParameterToProject(self):
        meanEachClassList, meanTotal = self._calculateMean()
        Sw = self._calculateWithinClassCovarianceMatrix(meanEachClassList)
        Sb = self._calculateBetweenClassCovarianceMatrix(meanEachClassList, meanTotal)
        
        eigVals, eigVecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        eigList = sorted([(eigVals[i], eigVecs[:, i]) for i in range(len(eigVals))], key=lambda x: x[0], reverse=True)
        
        self.W = np.hstack([eigList[i][1].reshape(-1, 1) for i in range(2)])
        return self.W

    def projectDataToTwoDimensions(self):
        self.dataMatrix = self.dataMatrix[:, 1:]
        self.W = self.findParameterToProject()
        dataInTwoDimensions = np.dot(self.dataMatrix, self.W)
        onesColumn = np.ones((dataInTwoDimensions.shape[0], 1))
        dataInTwoDimensions = np.hstack((onesColumn, dataInTwoDimensions))
        self.dataMatrix = dataInTwoDimensions
        return self.dataMatrix, self.target
    
    def projectInputToTwoDimensions(self, input):
        input = input[1:]
        self.dataMatrix = self.dataMatrix[:, 1:]
        input = np.dot(input, self.W)
        input = np.insert(input, 0, 1)
        return input
    
    def trainModel(self):
        self.projectDataToTwoDimensions()
        self.disFunc = ClassificFunction.discriminantFunction(self.targetMatrix, self.dataMatrix)

        return self.disFunc

    def predict(self, input):
        input = self.projectInputToTwoDimensions(input)

        prediction = self.disFunc(input)
        return np.argmax(prediction)
