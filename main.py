import numpy as np
from tools import Tools
from crossvalidator import CrossValidator

def main():
    filePath = 'Iris.csv'
    data, target = Tools.readCSV(filePath)
    classes = np.unique(target)

    crossValidator = CrossValidator(data, target, classes)
    
    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyThreeClass)
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix)

if __name__ == "__main__":
    main()