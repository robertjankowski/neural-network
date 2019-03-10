#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>

NeuralNet::NeuralNet(std::vector<int> s) : _sizes(s)
{
    _numLayers = _sizes.size();
    for (unsigned int i = 1; i < _sizes.size(); ++i)
    {
        int cols = _sizes.at(i);
        Matrix<double> bias(1, cols);
        bias.fillGauss(0, 1); // mean=0, variance=1
        _biases.push_back(bias);
    }
    std::vector<int> from(_sizes.begin() + 1, _sizes.end());
    std::vector<int> to(s.begin(), s.end() - 1);
    for (auto i : zip(to, from))
    {
        Matrix<double> weight(i.at(0), i.at(1));
        weight.fillGauss(0, 1); // mean=0, variance=1
        _weights.push_back(weight);
    }
}
Matrix<double> NeuralNet::feedforward(Matrix<double> a)
{
    // return output of network if `a` is input
    for (auto i : zip(_biases, _weights))
    {
        auto b = i.at(0);
        auto w = i.at(1);
        // for tests in other way
        auto dot = mul(a, w) + b;
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
        int nTest = testData.at(0).rows();
    }
    /**
     * trainingData = [ Matrix<double> inputs, Matrix<double> labels ]
     */
    int nTrain = trainingData.at(0).rows();
    for (int i = 0; i < epochs; ++i)
    {
        // shuffle training data
        auto inputMatrix = trainingData.at(0);
        auto labelsMatrix = trainingData.at(1);
        auto concat = inputMatrix.concatMatrix(labelsMatrix);
        auto shuffle = concat.shuffleRows();
        trainingData = shuffle.splitMatrix(inputMatrix.cols());

        // mini batches
        std::vector<std::vector<Matrix<double>>> miniBatches;
        inputMatrix = trainingData.at(0);
        labelsMatrix = trainingData.at(1);
        for (int k = 0; k < nTrain - miniBatchSize; k += miniBatchSize)
        {
            auto inputPartial = inputMatrix.getRows(k, k + miniBatchSize);
            auto labelsPartial = labelsMatrix.getRows(k, k + miniBatchSize);
            std::vector<Matrix<double>> vec = {inputPartial, labelsPartial};
            miniBatches.push_back(vec);
        }
        std::cout << "minibatch size: " << miniBatches.size() << '\n';
        for (auto miniBatch : miniBatches)
        {
            // TODO: finish update mini batch fuction
            updateMiniBatch(miniBatch, eta);
        }
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
    std::vector<Matrix<double>> nablaB = _biases;
    std::vector<Matrix<double>> nablaW = _weights;
    // fill zeros
    for (unsigned int i = 0; i < _biases.size(); ++i)
    {
        auto b = _biases.at(i);
        b.fill(0);
        nablaB.at(i) = b;
    }
    for (unsigned int i = 0; i < _weights.size(); ++i)
    {
        auto w = _weights.at(i);
        w.fill(0);
        nablaW.at(i) = w;
    }
    auto activation = x;
    // store all activations, layer by layer
    std::vector<Matrix<double>> activations = {activation};
    // store all the `z` vectors, layer by layer
    std::vector<Matrix<double>> zs;
    for (auto i : zip(_biases, _weights))
    {
        auto b = i.at(0), w = i.at(1);
        // TODO: ERROR: check sizes of matrices -> in different way
        auto z = mul<double>(activation, w) + b;
        zs.push_back(z);
        activation = applyActivation(Activation::sigmoid, z);
        activations.push_back(activation);
    }
    // backward pass
    auto delta = costDerivative(activations.at(activations.size() - 1), y);
    auto sigPrime = applyActivation(Activation::sigmoidDerivative,
                                    zs.at(zs.size() - 1));
    Matrix<double> deltaSigPrime(delta.rows(), delta.cols());
    for (int i = 0; i < delta.rows(); ++i)
    {
        auto deltaRow = delta.getOneRow(i);
        auto sigPrimeRow = sigPrime.getOneRow(i);
        std::vector<double> counter;
        for (unsigned int j = 0; j < deltaRow.size(); ++j)
        {
            counter.push_back(deltaRow.at(i) * sigPrimeRow.at(i));
        }
        deltaSigPrime.setOneRow(i, counter);
    }
    // auto deltaSigPrime = mul<double>(delta, sigPrime);
    nablaB.at(nablaB.size() - 1) = deltaSigPrime;
    auto forNablaW = activations.at(activations.size() - 2).transpose();
    nablaW.at(nablaW.size() - 1) = mul<double>(forNablaW, deltaSigPrime);

    for (int l = 2; l < _numLayers; ++l)
    {
        auto z = zs.at(zs.size() - l);
        auto sp = applyActivation(Activation::sigmoidDerivative, z);
        auto w = _weights.at(_weights.size() - l + 1).transpose();
        auto wMul = mul<double>(delta, w);

        // TODO: issuse with matrices sizes - there is `*` not `dot`
        // but `delta` is different size than `wMul` and `sp`
        delta = mul<double>(wMul, sp);
        nablaB.at(nablaB.size() - l + 1) = delta;
        activation = activations.at(activations.size() - l + 1).transpose();
        delta.showShape();
        activation.showShape();
        auto forNablaW = mul<double>(delta, activation);
        nablaW.at(nablaW.size() - l + 1) = mul<double>(delta, forNablaW);
    }
    // return
    std::vector<std::vector<Matrix<double>>> nablaVec = {nablaB, nablaW};
    return nablaVec;
}

void NeuralNet::updateMiniBatch(std::vector<Matrix<double>> miniBatch, double eta)
{
    std::vector<Matrix<double>> nablaB = _biases;
    std::vector<Matrix<double>> nablaW = _weights;
    // fill zeros
    for (unsigned int i = 0; i < _biases.size(); ++i)
    {
        auto b = _biases.at(i);
        b.fill(0);
        nablaB.at(i) = b;
    }
    for (unsigned int i = 0; i < _weights.size(); ++i)
    {
        auto w = _weights.at(i);
        w.fill(0);
        nablaW.at(i) = w;
    }
    auto features = miniBatch.at(0);
    auto labels = miniBatch.at(1);
    for (int i = 0; i < features.rows(); ++i)
    {
        Matrix<double> f(1, features.cols());
        f.setOneRow(0, features.getOneRow(i));
        Matrix<double> l(1, labels.cols());
        l.setOneRow(0, labels.getOneRow(i));

        // TODO: checking backprop function [error with multiplication]
        auto afterBackProp = backprop(f, l);
        auto deltaNablaB = afterBackProp.at(0);
        auto deltaNablaW = afterBackProp.at(1);

        std::vector<Matrix<double>> nablaBtmp;
        for (auto n : zip(nablaB, deltaNablaB))
        {
            auto nb = n.at(0);
            auto dnb = n.at(1);
            nablaBtmp.push_back(nb + dnb);
        }
        nablaB = nablaBtmp;

        std::vector<Matrix<double>> nablaWtmp;
        for (auto n : zip(nablaW, deltaNablaW))
        {
            auto nw = n.at(0);
            auto dnw = n.at(1);
            nablaWtmp.push_back(nw + dnw);
        }
        nablaW = nablaWtmp;
    }
    // update weights and biases
    std::vector<Matrix<double>> weightsTmp;
    for (auto i : zip(_weights, nablaW))
    {
        auto w = i.at(0);
        auto nw = i.at(1);
        auto stepTmp = nw * (eta / miniBatch.size());
        auto step = w - stepTmp;
        weightsTmp.push_back(step);
    }
    _weights = weightsTmp;

    std::vector<Matrix<double>> biasesTmp;
    for (auto i : zip(_biases, nablaB))
    {
        auto b = i.at(0);
        auto nb = i.at(1);
        auto stepTmp = nb * (eta / miniBatch.size());
        auto step = b - stepTmp;
        biasesTmp.push_back(step);
    }
    _biases = biasesTmp;
}