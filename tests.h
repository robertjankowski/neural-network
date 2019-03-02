#ifndef __TESTS__
#define __TESTS__

#include "loader.h"
#include "dataset.h"
#include "encoder.h"
#include "activation.h"
#include <iostream>
#include "matrix.h"
#include "neural_net.h"

void testRandom()
{
    // fill gausss
    Matrix<double> A(10, 10);
    A.fillGauss(2.0, 1.0);
    std::cout << "Fill gauss" << std::endl;
    std::cout << A << std::endl;

    // fill random
    Matrix<int> B(10, 10);
    B.fillRandom(100);
    std::cout << "Fill random:" << std::endl;
    std::cout << B << std::endl;

    // fill uniform 0-1
    Matrix<double> C(10, 10);
    C.fillUniform(0, 1);
    std::cout << "Fill uniform:" << std::endl;
    std::cout << C << std::endl;
}

void testActivationFunctions()
{
    auto N{10}, M{10};
    Matrix<double> X(N, M);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            X.set(i, j, 5 * 0.5 * ((double)rand() / RAND_MAX - 1));
        }
    }
    std::cout << X << std::endl;
    std::cout << "After sigmoid" << '\n';
    auto Y = applyActivation(Activation::sigmoid, X);
    std::cout << Y << std::endl;

    std::cout << "After tan hyberbolic" << '\n';
    auto Z = applyActivation(Activation::tanh, X);
    std::cout << Z << std::endl;
    auto Z1 = Z * 2;
    std::cout << Z1 << std::endl;
}

void testMatrixMultiplication()
{
    std::cout << "Testing matrix multiplication" << std::endl;
    Matrix<int> W1(3, 1);
    Matrix<int> W2(1, 3);
    W1.fill(3);
    W2.fill(12);
    std::cout << "W1:" << std::endl;
    W1.showShape();
    std::cout << W1 << std::endl;
    std::cout << "W2:" << std::endl;
    W2.showShape();
    std::cout << W2 << std::endl;
    auto W3 = mul<int>(W2, W1);
    std::cout << "After multiplication:" << std::endl;
    std::cout << W3 << std::endl;
}

void testTranspose()
{
    std::cout << "Testing transposing" << std::endl;
    Matrix<int> A(4, 10);
    A.fill(1);
    A.showShape();
    std::cout << A << std::endl;
    auto B = A.transpose();
    std::cout << B << std::endl;
    B.showShape();
}

void testRandomShuffle()
{
    std::cout << "Testing random shuffle matrix" << std::endl;
    Matrix<int> A(4, 4);
    A.fillRandom(10);
    std::cout << A << std::endl
              << "After shuffle:" << std::endl;
    A = A.shuffleRows();
    std::cout << A << std::endl;
}

void testMatrixConcatSplit()
{
    std::cout << "Testing matrix concat/split" << std::endl;
    Matrix<int> A(4, 2);
    A.fill(10);
    Matrix<int> B(4, 3);
    B.fill(4);
    auto C = A.concatMatrix(B);
    std::cout << C << std::endl;

    std::cout << "After split function" << std::endl;
    auto split = C.splitMatrix(2);
    std::cout << split.at(0) << std::endl;
    std::cout << split.at(1) << std::endl;
}

void testNeuralNet()
{
    std::cout << "Testing neural network" << std::endl;
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
    // 4 - input  neurons
    // 3 - output neurons
    std::vector<int> sizes = {4, 8, 3};
    NeuralNet nn(sizes);
    Matrix<double> inputMatrix(4, 1);
    inputMatrix.fillGauss(0, 1);
    nn.feedforward(inputMatrix);

    // create training data
    std::vector<Matrix<double>> trainingData;
    trainingData.push_back(input);
    trainingData.push_back(oneHotLabels);

    auto trainTestData = trainTestSplit(trainingData, 0.25);

    auto testData = trainTestData.at(0);
    auto trainData = trainTestData.at(1);

    int epochs = 10;
    int miniBatchSize = 5;
    double eta = 0.01;
    nn.SGD(trainData, epochs, miniBatchSize, eta, testData);
}

#endif // !__TESTS__
