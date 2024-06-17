import numpy as np
from file_reader import FileReader
from crossvalidator import CrossValidator

def main():
    filePath = 'data/Iris.csv'
    data, target = FileReader.readIrisCSV(filePath)
    classes = np.unique(target)

    crossValidator = CrossValidator(data, target, classes)
    
    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyFisher)
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix)

if __name__ == "__main__":
    main()