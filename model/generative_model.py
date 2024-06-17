import numpy as np
from model.model import Model

class GenerativeModel (Model):    
    def __init__(self, dataMatrix, target, classes):
        super().__init__(dataMatrix, target, classes)
        self.dataMatrix = self.dataMatrix[:, 1:]

    def trainModel(self):
        # Compute prior probabilities
        priorProVec = [len(self.dataMatrix[self.target == cls]) / len(self.dataMatrix) for cls in self.classes]
        
        # Compute mean distribution for each class
        meanDistrVec = [np.mean(self.dataMatrix[self.target == cls], axis=0) for cls in self.classes]
        
        # Compute covariance for each class
        covarianceEachClassVec = []
        for i, cls in enumerate(self.classes):
            dataCls = self.dataMatrix[self.target == cls]
            covarianceCls = np.cov(dataCls, rowvar=False)
            covarianceEachClassVec.append(covarianceCls)
        
        # Compute shared covariance matrix
        sharedCovariance = sum(covarianceEachClassVec) / len(self.classes)
        
        # Compute weights
        invSharedCovariance = np.linalg.inv(sharedCovariance)
        self.w = np.dot(invSharedCovariance, np.array(meanDistrVec).T)
        
        # Compute w0
        self.w0 = []
        for i in range(len(self.classes)):
            meanCls = meanDistrVec[i]
            logPrior = np.log(priorProVec[i])
            term = -0.5 * np.dot(np.dot(meanCls.T, invSharedCovariance), meanCls)
            self.w0.append(term + logPrior)
        
        self.w0 = np.array(self.w0)

    def predict(self, input):
        input = np.array(input[1:])
        a = np.dot(self.w.T, input) + self.w0
        
        expA = np.exp(a)
        probability = expA / np.sum(expA)
        
        return np.argmax(probability)

