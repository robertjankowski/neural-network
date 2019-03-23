#include "loader.h"
#include "dataset.h"
#include "encoder.h"
#include "activation.h"
#include <iostream>
#include "matrix.h"
#include "neural_net.h"
#include <iomanip>

int main()
{
    // TODO: add parameters to script, e.g. to select learning rate and epoches

    Loader l("iris.data", ',');
    auto input = l.getInput();
    auto classes = l.getClasses();
    // encode string to int
    Encoder e;
    e.fit(classes);
    auto labels = e.getLabels();
    // classes to OneHotEncoder
    OneHotEncoder oneHot;
    oneHot.toOneHot(labels);
    auto oneHotLabels = oneHot.getOneHot().convertToDouble();

    // build NN
    std::vector<int> sizes = {4, 8, 3};
    NeuralNet nn(sizes);

    auto data = convertData(input.transpose(), oneHotLabels.transpose());
    // train/test split
    auto trainTest = trainTestSplit(data, 0.2);
    auto trainData = std::get<0>(trainTest);
    auto testData = std::get<1>(trainTest);

    int epochs = 100;
    int miniBatchSize = 10;
    double eta = 0.05;
    nn.SGD(trainData, epochs, miniBatchSize, eta, testData);

    double accuracy = nn.accuracy(testData);
    std::cout << std::setprecision(4);
    std::cout << "Accuracy: " << accuracy << std::endl;
}
