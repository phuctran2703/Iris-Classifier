import numpy as np
from model.model import Model
from discriminant_function import ClassificFunction

class OneVersusOneModel(Model):
    def __init__(self, dataMatrix, target, classes):
        super().__init__(dataMatrix, target, classes)

    def classifyTwoClasses(self, class1, class2):
        # Filter the dataset for the two classes
        idx = np.where((self.target == class1) | (self.target == class2))[0]
        dataSubsetMatrix = self.dataMatrix[idx]
        targetSubset = self.target[idx]
        
        # One-of-K encoding for target
        targetSubsetMatrix = np.zeros((targetSubset.shape[0], 2))
        targetSubsetMatrix[targetSubset == class1, 0] = 1
        targetSubsetMatrix[targetSubset == class2, 1] = 1
        
        # Train the discriminant function
        f = ClassificFunction.discriminantFunction(targetSubsetMatrix, dataSubsetMatrix)
        
        return f
    
    def trainModel(self):
        self.class1vsclass2DiscFunc = self.classifyTwoClasses(self.classes[0], self.classes[1])
        self.class1vsclass3DiscFunc = self.classifyTwoClasses(self.classes[0], self.classes[2])
        self.class2vsclass3DiscFunc = self.classifyTwoClasses(self.classes[1], self.classes[2])


    def predict(self, input):
        predictions = []
        predictions1 = self.class1vsclass2DiscFunc(input)
        predictions2 = self.class1vsclass3DiscFunc(input)
        predictions3 = self.class2vsclass3DiscFunc(input)
        
        if(predictions1[0] > predictions1[1]): predictions.append(0)
        else: predictions.append(1)
        if(predictions2[0] > predictions2[1]): predictions.append(0)
        else: predictions.append(2)
        if(predictions3[0] > predictions3[1]): predictions.append(1)
        else: predictions.append(2)
        
        count0 = np.count_nonzero(np.array(predictions) == 0)
        count1 = np.count_nonzero(np.array(predictions) == 1)
        count2 = np.count_nonzero(np.array(predictions) == 2)

        if count0 > count1 and count0 > count2: return 0
        elif count1 > count0 and count1 > count2: return 1
        elif count2 > count0 and count2 > count1: return 2

        return 0
