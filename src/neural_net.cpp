#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>
#include <cmath>
#include <random>

NeuralNet::NeuralNet(std::vector<int> s) : _sizes(s)
{
    _numLayers = _sizes.size();
    std::vector<int> from(_sizes.begin() + 1, _sizes.end());
    std::vector<int> to(s.begin(), s.end() - 1);
    for (auto i : zip(to, from))
    {
        Matrix<double> weight(i.at(1), i.at(0));
        weight.fillGauss(0, 1);
        _weights.push_back(weight);
    }

    for (unsigned int i = 1; i < _sizes.size(); ++i)
    {
        int cols = _sizes.at(i);
        Matrix<double> bias(cols, 1);
        bias.fillGauss(0, 1);
        _biases.push_back(bias);
    }
}
Matrix<double> NeuralNet::feedforward(Matrix<double> a)
{
    for (auto i : zip(_biases, _weights))
    {
        auto b = i.at(0);
        auto w = i.at(1);
        auto dot = mul(w, a) + b;
        a = applyActivation(Activation::sigmoid, dot);
    }
    return a;
}

int NeuralNet::predict(Matrix<double> &input)
{
    auto output = feedforward(input);
    int argMax = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    return argMax;
}

double NeuralNet::accuracy(dataVector<double> &testData)
{
    double accur = 0.0;
    for (unsigned int i = 0; i < testData.size(); ++i)
    {
        auto X = testData.at(i).at(0);
        auto y = testData.at(i).at(1);
        int yArg = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        int pred = predict(X);
        if (yArg == pred)
            accur += 1;
    }
    return accur / testData.size();
}

double NeuralNet::loss(Matrix<double> &yTrue, Matrix<double> &yPred)
{
    // mean squared error - MSE
    double l;
    if (yTrue.rows() != yPred.rows())
        std::cerr << "Wrong matrices shape" << std::endl;
    for (int i = 0; i < yTrue.rows(); ++i)
        l += pow(yTrue.at(i, 0) - yPred.at(i, 0), 2);
    return 1.0 / (2 * yTrue.rows()) * sqrt(l);
}

void NeuralNet::SGD(dataVector<double> &trainData, int epochs, int miniBatchSize, double eta,
                    dataVector<double> &testData)
{

    for (int i = 0; i < epochs; ++i)
    {
        shuffleData(trainData);
        auto miniBatches = splitIntoMiniBatches(trainData, miniBatchSize);
        for (auto &batch : miniBatches)
            updateMiniBatch(batch, eta);

        double l = 0.0;
        for (auto &test : testData)
        {
            auto X_test = test.at(0);
            auto y_test = test.at(1);
            auto output = feedforward(X_test);
            l += loss(y_test, output) / testData.size();
        }
        std::cout << "Epoch: " << i + 1 << "/" << epochs << " complete\tTest loss: " << l << std::endl;
    }
}

void NeuralNet::updateMiniBatch(dataVector<double> &batch, double eta)
{
    auto nabla_b = fillZeros(_biases);
    auto nabla_w = fillZeros(_weights);

    for (unsigned int i = 0; i < batch.size(); ++i)
    {
        auto X = batch.at(i).at(0);
        auto y = batch.at(i).at(1);

        auto afterBackProp = backprop(X, y);
        auto delta_nabla_b = afterBackProp.first;
        auto delta_nabla_w = afterBackProp.second;

        for (unsigned int i = 0; i < nabla_b.size(); ++i)
        {
            auto nb = nabla_b.at(i);
            auto dnb = delta_nabla_b.at(i);
            nabla_b.at(i) = nb + dnb;
        }
        for (unsigned int i = 0; i < nabla_w.size(); ++i)
        {
            auto nw = nabla_w.at(i);
            auto dnw = delta_nabla_w.at(i);
            nabla_w.at(i) = nw + dnw;
        }
    }
    updateWeightsAndBiases(nabla_b, nabla_w, batch.size(), eta);
}

void NeuralNet::updateWeightsAndBiases(matrixVector<double> &nabla_b, matrixVector<double> &nabla_w,
                                       unsigned int batchSize, double eta)
{
    for (unsigned int i = 0; i < _weights.size(); ++i)
    {
        auto w = _weights.at(i);
        auto nw = nabla_w.at(i);
        auto d = nw * (eta / batchSize);
        _weights.at(i) = w - d;
    }
    for (unsigned int i = 0; i < _biases.size(); ++i)
    {
        auto b = _biases.at(i);
        auto nb = nabla_b.at(i);
        auto d = nb * (eta / batchSize);
        _biases.at(i) = b - d;
    }
}

pair<double> NeuralNet::backprop(Matrix<double> &X, Matrix<double> &y)
{
    auto nabla_b = fillZeros(_biases);
    auto nabla_w = fillZeros(_weights);

    // feedforward
    auto activation = X;
    matrixVector<double> activations = {X};
    matrixVector<double> zs;
    feedforward(activations, activation, zs);

    backwardPass(nabla_b, nabla_w, activations, zs, y);
    return std::make_pair(nabla_b, nabla_w);
}

void NeuralNet::feedforward(matrixVector<double> &activations, Matrix<double> &activation, matrixVector<double> &zs)
{
    for (auto i : zip(_biases, _weights))
    {
        auto b = i.at(0);
        auto w = i.at(1);

        auto z = mul(w, activation) + b;
        zs.push_back(z);
        activation = applyActivation(Activation::sigmoid, z);
        activations.push_back(activation);
    }
}

void NeuralNet::backwardPass(matrixVector<double> &nabla_b, matrixVector<double> &nabla_w,
                             matrixVector<double> &activations, matrixVector<double> &zs, Matrix<double> &y)
{
    auto delta = dot(costDerivative(activations.at(activations.size() - 1), y),
                     applyActivation(Activation::sigmoidDerivative, zs.at(zs.size() - 1)));

    nabla_b.at(nabla_b.size() - 1) = delta;
    nabla_w.at(nabla_w.size() - 1) = mul(delta, activations.at(activations.size() - 2).transpose());

    for (int l = 2; l < _numLayers; ++l)
    {
        auto z = zs.at(zs.size() - l);
        auto sp = applyActivation(Activation::sigmoidDerivative, z);
        delta = dot(mul(_weights.at(_weights.size() - l + 1).transpose(), delta), sp);
        nabla_b.at(nabla_b.size() - l) = delta;
        nabla_w.at(nabla_w.size() - l) = mul(delta, activations.at(activations.size() - l - 1).transpose());
    }
}

Matrix<double> NeuralNet::costDerivative(Matrix<double> outputActivation, Matrix<double> y)
{
    /**
     * return the vector of partial derivatives
     * \partial C_x / \partial a
     */
    return (outputActivation - y);
}

Matrix<double> NeuralNet::confusionMatrix(dataVector<double> testData)
{
    int shape = testData.at(0).at(1).rows();
    Matrix<double> matrix(shape, shape);
    matrix.fill(0);

    for (unsigned int i = 0; i < testData.size(); ++i)
    {
        auto X = testData.at(i).at(0);
        auto y = testData.at(i).at(1);
        int yTrue = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        int yPred = predict(X);

        // TODO: very wrong way, refactor it later
        if (yTrue == 0 && yPred == 0)
            matrix.set(0, 0, matrix.at(0, 0) + 1);
        else if (yTrue == 1 && yPred == 1)
            matrix.set(1, 1, matrix.at(1, 1) + 1);
        else if (yTrue == 2 && yPred == 2)
            matrix.set(2, 2, matrix.at(2, 2) + 1);
        else if (yTrue == 0 && yPred == 1)
            matrix.set(0, 1, matrix.at(0, 1) + 1);
        else if (yTrue == 1 && yPred == 0)
            matrix.set(1, 0, matrix.at(1, 0) + 1);
        else if (yTrue == 2 && yPred == 0)
            matrix.set(2, 0, matrix.at(2, 0) + 1);
        else if (yTrue == 0 && yPred == 2)
            matrix.set(0, 2, matrix.at(0, 2) + 1);
        else if (yTrue == 1 && yPred == 2)
            matrix.set(1, 2, matrix.at(1, 2) + 1);
        else if (yTrue == 2 && yPred == 1)
            matrix.set(2, 1, matrix.at(2, 1) + 1);
    }
    return matrix;
}

dataVector<double> convertData(Matrix<double> X, Matrix<double> y)
{
    /**
     * Input: X shape : (features, no. of instances)
     *        y shape : (labels,   no. of instances)
     * 
     * Output: vector < vector < X', y' > >
     *    where X' shape : (features, 1)
     *          y' shape : (labels,   1)
     */
    dataVector<double> data;
    for (int i = 0; i < X.cols(); ++i)
    {
        matrixVector<double> v = {X.getOneCol(i), y.getOneCol(i)};
        data.push_back(v);
    }
    return data;
}

void shuffleData(dataVector<double> &X)
{
    std::srand(std::time(nullptr));
    auto engine = std::default_random_engine{};
    std::shuffle(X.begin(), X.end(), engine);
}

miniBatchVector<double> splitIntoMiniBatches(dataVector<double> &X, int miniBatchSize)
{
    miniBatchVector<double> miniBatches;
    for (unsigned int i = 0; i < X.size(); i += miniBatchSize)
    {
        dataVector<double> batch(X.begin() + i, X.begin() + i + miniBatchSize);
        miniBatches.push_back(batch);
    }
    return miniBatches;
}

std::pair<dataVector<double>, dataVector<double>> trainTestSplit(dataVector<double> &data, double testSize)
{
    unsigned int testCol = (int)(data.size() * testSize);
    dataVector<double> test;
    dataVector<double> train;

    shuffleData(data);
    for (unsigned int i = 0; i < data.size(); ++i)
    {
        if (i < testCol)
            test.push_back(data.at(i));
        else
            train.push_back(data.at(i));
    }
    return std::make_pair(train, test);
}

matrixVector<double> fillZeros(matrixVector<double> &X)
{
    matrixVector<double> zeros = X;
    for (auto &z : zeros)
        z.fill(0);
    return zeros;
}