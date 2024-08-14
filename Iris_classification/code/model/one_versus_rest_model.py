import numpy as np
from model.model import Model
from discriminant_function import ClassificFunction

class OneVersusRestModel(Model):
    def __init__(self, dataMatrix, target, classes):
        super().__init__(dataMatrix, target, classes)

    def classifyOneClasses(self, class1):        
        # One-of-K encoding for target
        targetMatrix = np.zeros((self.target.shape[0], 2))
        targetMatrix[self.target == class1, 0] = 1
        targetMatrix[self.target != class1, 1] = 1
        
        # Train the discriminant function
        f = ClassificFunction.discriminantFunction(targetMatrix, self.dataMatrix)
        
        return f
    
    def trainModel(self):
        self.class1DiscFunc = self.classifyOneClasses(self.classes[0])
        self.class2DiscFunc = self.classifyOneClasses(self.classes[1])


    def predict(self, input):
        predictions1 = self.class1DiscFunc(input)
        predictions2 = self.class2DiscFunc(input)
            
        if(predictions1[0] > predictions1[1] and predictions2[0] < predictions2[1]): return 0
        elif(predictions1[0] < predictions1[1] and predictions2[0] > predictions2[1]): return 1
        elif(predictions1[0] < predictions1[1] and predictions2[0] < predictions2[1]): return 2

        return 0
