#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>
#include <cmath>
#include <random>

std::vector<std::vector<Matrix<double>>> convertData(Matrix<double>, Matrix<double>);
void shuffleData(std::vector<std::vector<Matrix<double>>> &);
std::vector<std::vector<std::vector<Matrix<double>>>> splitIntoMiniBatches(std::vector<std::vector<Matrix<double>>> &, int);

NeuralNet::NeuralNet(std::vector<int> s) : _sizes(s)
{
    _numLayers = _sizes.size();
    std::vector<int> from(_sizes.begin() + 1, _sizes.end());
    std::vector<int> to(s.begin(), s.end() - 1);
    for (auto i : zip(to, from))
    {
        Matrix<double> weight(i.at(1), i.at(0));
        weight.fillGauss(0, 1); // mean=0, variance=1
        _weights.push_back(weight);
    }

    for (unsigned int i = 1; i < _sizes.size(); ++i)
    {
        int cols = _sizes.at(i);
        Matrix<double> bias(cols, 1);
        bias.fillGauss(0, 1); // mean = 0, variance = 1
        _biases.push_back(bias);
    }
}
Matrix<double> NeuralNet::feedforward(Matrix<double> a)
{
    // return output of network if `a` is input
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

double NeuralNet::accuracy(std::vector<std::vector<Matrix<double>>> testData)
{
    double accur = 0.0;
    for (int i = 0; i < testData.size(); ++i)
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
    {
        std::cerr << "Wrong matrices shape" << std::endl;
    }
    for (int i = 0; i < yTrue.rows(); ++i)
    {
        l += pow(yTrue.at(i, 0) - yPred.at(i, 0), 2);
    }
    return 1.0 / (2 * yTrue.rows()) * sqrt(l);
}

void NeuralNet::SGD(std::vector<std::vector<Matrix<double>>> trainData, int epochs,
                    int miniBatchSize, double eta, std::vector<std::vector<Matrix<double>>> testData)
{

    for (int i = 0; i < epochs; ++i)
    {
        // shuffle training data
        shuffleData(trainData);
        // split into mini batches
        auto miniBatches = splitIntoMiniBatches(trainData, miniBatchSize);
        for (auto &batch : miniBatches)
        {
            // update_mini_batch
            updateMiniBatch(batch, eta);
        }
        // calculate loss
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

void NeuralNet::updateMiniBatch(std::vector<std::vector<Matrix<double>>> &batch, double eta)
{
    auto nabla_b = _biases;
    auto nabla_w = _weights;
    // fill zero
    for (auto &b : nabla_b)
    {
        b.fill(0);
    }
    for (auto &w : nabla_w)
    {
        w.fill(0);
    }

    for (unsigned int i = 0; i < batch.size(); ++i)
    {
        auto X = batch.at(i).at(0);
        auto y = batch.at(i).at(1);

        auto afterBackProp = backprop(X, y);
        auto delta_nabla_b = std::get<0>(afterBackProp);
        auto delta_nabla_w = std::get<1>(afterBackProp);

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

    // update weights and biases
    for (unsigned int i = 0; i < _weights.size(); ++i)
    {
        auto w = _weights.at(i);
        auto nw = nabla_w.at(i);
        auto d = nw * (eta / batch.size());
        _weights.at(i) = w - d;
    }
    for (unsigned int i = 0; i < _biases.size(); ++i)
    {
        auto b = _biases.at(i);
        auto nb = nabla_b.at(i);
        auto d = nb * (eta / batch.size());
        _biases.at(i) = b - d;
    }
}

std::tuple<std::vector<Matrix<double>>, std::vector<Matrix<double>>> NeuralNet::backprop(Matrix<double> X, Matrix<double> y)
{
    auto nabla_b = _biases;
    auto nabla_w = _weights;
    // fill zero
    for (auto &b : nabla_b)
    {
        b.fill(0);
    }
    for (auto &w : nabla_w)
    {
        w.fill(0);
    }

    // feedforward
    auto activation = X;
    std::vector<Matrix<double>> activations = {X};
    std::vector<Matrix<double>> zs;

    for (auto i : zip(_biases, _weights))
    {
        auto b = i.at(0);
        auto w = i.at(1);

        auto z = mul(w, activation) + b;
        zs.push_back(z);
        activation = applyActivation(Activation::sigmoid, z);
        activations.push_back(activation);
    }

    // backward pass
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
    return std::make_tuple(nabla_b, nabla_w);
}

Matrix<double> NeuralNet::costDerivative(Matrix<double> outputActivation, Matrix<double> y)
{
    /**
     * return the vector of partial derivatives
     * \partial C_x / \partial a
     */
    return (outputActivation - y);
}

std::vector<std::vector<Matrix<double>>> convertData(Matrix<double> X, Matrix<double> y)
{
    /**
     * Input: X shape : (features, no. of instances)
     *        y shape : (labels,   no. of instances)
     * 
     * Output: vector < vector < X', y' > >
     *    where X' shape : (features, 1)
     *          y' shape : (labels,   1)
     */
    std::vector<std::vector<Matrix<double>>> data;
    for (int i = 0; i < X.cols(); ++i)
    {
        std::vector<Matrix<double>> v = {X.getOneCol(i), y.getOneCol(i)};
        data.push_back(v);
    }
    return data;
}

void shuffleData(std::vector<std::vector<Matrix<double>>> &X)
{
    std::srand(std::time(nullptr));
    auto engine = std::default_random_engine{};
    std::shuffle(X.begin(), X.end(), engine);
}

std::vector<std::vector<std::vector<Matrix<double>>>> splitIntoMiniBatches(std::vector<std::vector<Matrix<double>>> &X,
                                                                           int miniBatchSize)
{
    std::vector<std::vector<std::vector<Matrix<double>>>> miniBatches;
    for (unsigned int i = 0; i < X.size(); i += miniBatchSize)
    {
        std::vector<std::vector<Matrix<double>>> batch(X.begin() + i, X.begin() + i + miniBatchSize);
        miniBatches.push_back(batch);
    }
    return miniBatches;
}