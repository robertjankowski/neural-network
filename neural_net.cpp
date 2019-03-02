#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>

NeuralNet::NeuralNet(std::vector<int> s) : sizes(s)
{
    numLayers = sizes.size();
    for (unsigned int i = 1; i < sizes.size(); ++i)
    {
        int rows = sizes.at(i);
        Matrix<double> bias(rows, 1);
        bias.fillGauss(0, 1); // mean=0, variance=1
        biases.push_back(bias);
    }
    std::vector<int> from(sizes.begin() + 1, sizes.end());
    std::vector<int> to(s.begin(), s.end() - 1);
    for (auto i : zip(from, to))
    {
        Matrix<double> weight(i.at(0), i.at(1));
        weight.fillGauss(0, 1); // mean=0, variance=1
        weights.push_back(weight);
    }
}
Matrix<double> NeuralNet::feedforward(Matrix<double> a)
{
    // return output of network if `a` is input
    for (auto i : zip(biases, weights))
    {
        auto b = i.at(0);
        auto w = i.at(1);
        auto dot = mul(w, a) + b;
        a = applyActivation(Activation::sigmoid, dot);
    }
    return a;
}

void NeuralNet::SGD(std::vector<Matrix<double>> trainingData, int epochs,
                    int miniBatchSize, double eta, std::vector<Matrix<double>> testData)
{
    std::cout << "Beginning stochastic gradient descent..." << '\n';
    if (!testData.empty())
    {
        int nTest = testData.size();
    }
    /**
     * trainingData = [ Matrix<double> inputs, Matrix<double> labels ]
     */
    int nTrain = trainingData.size();
    for (int i = 0; i < epochs; ++i)
    {
        // shuffle training data
        auto inputMatrix = trainingData.at(0);
        auto labelsMatrix = trainingData.at(1);
        auto concat = inputMatrix.concatMatrix(labelsMatrix);
        auto shuffle = concat.shuffleRows();
        trainingData = shuffle.splitMatrix(inputMatrix.cols());

        // mini batches
        // TODO: http://neuralnetworksanddeeplearning.com/chap1.html

        std::cout << "Epoch " << i << " complete" << std::endl;
    }
}

Matrix<double> NeuralNet::costDerivative(Matrix<double> outputActivation,
                                         Matrix<double> y)
{
    return (outputActivation - y);
}

std::vector<std::vector<Matrix<double>>> NeuralNet::backprop(Matrix<double> x,
                                                             Matrix<double> y)
{
    std::vector<Matrix<double>> nablaB = biases;
    std::vector<Matrix<double>> nablaW = weights;

    // fill zeros
    for (unsigned int i = 0; i < biases.size(); ++i)
    {
        auto b = biases.at(i);
        b.fill(0);
        nablaB.at(i) = b;
    }
    for (unsigned int i = 0; i < weights.size(); ++i)
    {
        auto w = weights.at(i);
        w.fill(0);
        nablaW.at(i) = w;
    }
    auto activation = x;
    // store all activations, layer by layer
    std::vector<Matrix<double>> activations = {activation};
    // store all the `z` vectors, layer by layer
    std::vector<Matrix<double>> zs;
    for (auto i : zip(biases, weights))
    {
        auto b = i.at(0), w = i.at(1);
        auto z = mul(w, activation) + b;
        zs.push_back(z);
        activation = applyActivation(Activation::sigmoid, z);
        activations.push_back(activation);
    }
    // backward pass
    auto delta = costDerivative(activations.at(activations.size() - 1), y);
    auto sigPrime = applyActivation(Activation::sigmoidDerivative,
                                    zs.at(zs.size() - 1));
    auto deltaSigPrime = mul(delta, sigPrime);
    nablaB.at(nablaB.size() - 1) = deltaSigPrime;
    auto forNablaW = activations.at(activations.size() - 2).transpose();
    nablaW.at(nablaW.size() - 1) = mul(deltaSigPrime, forNablaW);

    for (int l = 2; l < numLayers; ++l)
    {
        auto z = zs.at(zs.size() - l);
        auto sp = applyActivation(Activation::sigmoidDerivative, z);
        // TODO: complete for loop!
    }

    // return
    std::vector<std::vector<Matrix<double>>> nablaVec = {nablaB, nablaW};
    return nablaVec;
}