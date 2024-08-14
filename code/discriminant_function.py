import numpy as np

class ClassificFunction:
    # Discriminant function for classification
    @staticmethod
    def discriminantFunction(targetMatrix, dataMatrix):
        dataPseudoInverseMatrix = np.linalg.pinv(dataMatrix)
        paraMatrix = np.dot(dataPseudoInverseMatrix, targetMatrix)
        return lambda x: np.dot(paraMatrix.T, x)
    