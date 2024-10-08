import pandas as pd
import numpy as np

class FileReader:
    # Read CSV
    @staticmethod
    def readIrisCSV(filePath="Iris.csv"):
        dataFrame = pd.read_csv(filePath)
        irisData = dataFrame.drop(['Id', 'Species'], axis=1).values / 10

        # Add 1 into each row in matrix
        onesColumn = np.ones((irisData.shape[0], 1))
        irisData = np.hstack((onesColumn, irisData))

        irisTarget = dataFrame['Species'].values
        nameClasses = np.unique(irisTarget)
        for i in range(len(nameClasses)):
            irisTarget = np.where(irisTarget == nameClasses[0], 0, irisTarget)
            irisTarget = np.where(irisTarget == nameClasses[1], 1, irisTarget)
            irisTarget = np.where(irisTarget == nameClasses[2], 2, irisTarget)
        return irisData, irisTarget

    
    