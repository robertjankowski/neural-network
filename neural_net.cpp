#include "neural_net.h"
#include "activation.h"
#include <iostream>
#include <vector>
#include "zip.h"
#include <algorithm>

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

int NeuralNet::predict(Matrix<double> input)
{
    auto output = feedforward(input);
    int argMax = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    return argMax;
}
