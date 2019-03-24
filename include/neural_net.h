#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include "matrix.h"
#include <vector>
#include <tuple>
#include <utility>

// using own pair
template <class T>
using pair = std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>>;

class NeuralNet
{
    /**
     * ``_sizes`` contains numbers of neurons in each layer
     *  first layer is assumed to be input layer (is without bias) 
     *  the last layer - output
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
    double loss(Matrix<double> &, Matrix<double> &);
    void SGD(std::vector<std::vector<Matrix<double>>>, int, int, double, std::vector<std::vector<Matrix<double>>>);
    int predict(Matrix<double> &);
    double accuracy(std::vector<std::vector<Matrix<double>>>);
    void updateMiniBatch(std::vector<std::vector<Matrix<double>>> &, double);
    pair<double> backprop(Matrix<double>, Matrix<double>);
    Matrix<double> costDerivative(Matrix<double>, Matrix<double>);
    Matrix<double> confusionMatrix(std::vector<std::vector<Matrix<double>>>);
};

std::vector<std::vector<Matrix<double>>> convertData(Matrix<double>, Matrix<double>);
void shuffleData(std::vector<std::vector<Matrix<double>>> &);
std::vector<std::vector<std::vector<Matrix<double>>>> splitIntoMiniBatches(std::vector<std::vector<Matrix<double>>> &, int);

#endif // !__NEURAL_NET__
