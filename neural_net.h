#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include "matrix.h"
#include <vector>

class NeuralNet
{
    /**
     * ``_sizes`` contains numbers of neurons in each layer
     *  first layer is assumed to be input layer (is without bias)
    **/
    int _numLayers;
    std::vector<int> _sizes;
    std::vector<Matrix<double>> _weights;
    std::vector<Matrix<double>> _biases;

  public:
    NeuralNet(std::vector<int>);
    Matrix<double> feedforward(Matrix<double>);
    void SGD(std::vector<Matrix<double>>, int, int, double,
             std::vector<Matrix<double>> = std::vector<Matrix<double>>());
    Matrix<double> costDerivative(Matrix<double>, Matrix<double>);
    std::vector<std::vector<Matrix<double>>> backprop(Matrix<double>, Matrix<double>);
    void updateMiniBatch(std::vector<Matrix<double>>, double);
};

#endif // !__NEURAL_NET__
