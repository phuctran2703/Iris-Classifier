import numpy as np
from model.model import Model
from discriminant_function import ClassificFunction

class MulticlassModel(Model):
    def __init__(self, dataMatrix, target, classes):
        super().__init__(dataMatrix, target, classes)
    
    def trainModel(self):
        self.disFunc = ClassificFunction.discriminantFunction(self.targetMatrix, self.dataMatrix)

        return self.disFunc


    def predict(self, input):
        prediction = self.disFunc(input)
        return np.argmax(prediction)
