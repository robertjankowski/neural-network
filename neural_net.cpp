#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>
#include <cmath>
#include <random>

std::vector<std::vector<Matrix<double>>> convertData(Matrix<double>, Matrix<double>);

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
    return sqrt(l);
}

void NeuralNet::SGD(std::vector<std::vector<Matrix<double>>> trainData, int epochs,
                    int miniBatchSize, double eta, std::vector<std::vector<Matrix<double>>> testData)
{

    for (int i = 0; i < epochs; ++i)
    {
        // shuffle training data

        // create minibatches

        // for m in minibatches: update_mini_batch

        // print loss

        std::cout << "Epoch: " << i << "/" << epochs << " complete" << std::endl;
    }
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
