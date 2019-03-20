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
    std::vector<Matrix<double>> getBiases() { return _biases; }
    std::vector<Matrix<double>> getWeights() { return _weights; }
    Matrix<double> feedforward(Matrix<double>);

    int predict(Matrix<double>);
};

#endif // !__NEURAL_NET__
