#include "loader.h"
#include "dataset.h"
#include "encoder.h"
#include "activation.h"
#include <iostream>
#include "matrix.h"
#include "neural_net.h"
#include <iomanip>
#include <string>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: <neuralnet> <epochs> <minibatchsize> <learning_rate>" << std::endl;
        return 1;
    }

    Loader l("iris.data", ',');
    auto input = l.getInput();
    auto classes = l.getClasses();
    // encode string to int
    Encoder e;
    e.fit(classes);
    auto labels = e.getLabels();

    OneHotEncoder oneHot;
    oneHot.toOneHot(labels);
    auto oneHotLabels = oneHot.getOneHot().convertToDouble();

    std::vector<int> sizes = {4, 8, 3};
    NeuralNet nn(sizes);

    auto data = convertData(input.transpose(), oneHotLabels.transpose());
    auto trainTest = trainTestSplit(data, 0.2);
    auto trainData = trainTest.first;
    auto testData = trainTest.second;

    int epochs = std::stoi(argv[1]);
    int miniBatchSize = std::stoi(argv[2]);
    double eta = std::stod(argv[3]);
    nn.SGD(trainData, epochs, miniBatchSize, eta, testData);

    double accuracy = nn.accuracy(testData);
    std::cout << std::setprecision(4);
    std::cout << "\nAccuracy: " << accuracy << std::endl;

    auto confusionMatrix = nn.confusionMatrix(testData);
    std::cout << "\nConfunsion matrix" << std::endl;
    std::cout << confusionMatrix << std::endl;
}
