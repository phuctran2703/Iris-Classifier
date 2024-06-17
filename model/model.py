from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    def __init__(self, dataMatrix, target, classes):
        self.dataMatrix = np.array(dataMatrix)
        self.target = np.array(target)
        self.classes = np.array(classes)
        self.targetMatrix = np.zeros((target.shape[0], len(classes)))
        
        for i in range(len(classes)):
            self.targetMatrix[self.target == self.classes[i], i] = 1
    
    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def predict(self, input):
        pass