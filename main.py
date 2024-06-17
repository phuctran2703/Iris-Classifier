import numpy as np
from file_reader import FileReader
from crossvalidator import CrossValidator

def main():
    filePath = 'data/Iris.csv'
    data, target = FileReader.readIrisCSV(filePath)
    classes = np.unique(target)

    crossValidator = CrossValidator(data, target, classes)
    
    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyOneVersusOne)
    print('Classify One Versus One')
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix,"\n\n")

    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyOneVersusRest)
    print('Classify One Versus Rest')
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix,"\n\n")

    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyThreeClass)
    print('Classify Three Class')
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix,"\n\n")

    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyGeneration)
    print('Classify by Generation Model')
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix,"\n\n")

    averageAccuracy, confusionMatrix = crossValidator.crossValidate(crossValidator.classifyFisher)
    print('Classify by Fisher')
    print(f'Average Accuracy: {averageAccuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(confusionMatrix,"\n\n")

if __name__ == "__main__":
    main()