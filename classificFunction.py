import pandas as pd
import numpy as np
from tools import Tools

class ClassificFunction:
    # One-versus-One Classifier
    @staticmethod
    def oneVersusOne(data, target, class1, class2):
        # Filter the dataset for the two classes
        idx = np.where((target == class1) | (target == class2))[0]
        dataSubset = data[idx]
        targetSubset = target[idx]
        
        # One-of-K encoding for target
        targetMatrix = np.zeros((targetSubset.shape[0], 2))
        targetMatrix[targetSubset == class1, 0] = 1
        targetMatrix[targetSubset == class2, 1] = 1
        
        # Train the discriminant function
        f = Tools.discriminantFunction(targetMatrix, dataSubset)
        
        return f
    
    # One-versus-the-rest Classifier
    @staticmethod
    def oneVersusRest(data, target, class1):
        dataMatrix = np.array(data)

        # One-of-K encoding for target
        targetArray = np.array(target)
        targetMatrix = np.zeros((targetArray.shape[0],2))
        targetMatrix[targetArray == class1, 0] = 1
        targetMatrix[targetArray != class1, 1] = 1
        
        # Train the discriminant function
        f = Tools.discriminantFunction(targetMatrix, dataMatrix)

        return f
    
    # 3-class discriminant classifier 
    @staticmethod
    def threeClassDiscriminate(data, target, class1, class2, class3):
        dataMatrix = np.array(data)

        # One-of-K encoding for target
        targetArray = np.array(target)
        targetMatrix = np.zeros((targetArray.shape[0],3))
        targetMatrix[targetArray == class1, 0] = 1
        targetMatrix[targetArray == class2, 1] = 1
        targetMatrix[targetArray == class3, 2] = 1
        
        # Train the discriminant function
        f = Tools.discriminantFunction(targetMatrix, dataMatrix)

        return f
        