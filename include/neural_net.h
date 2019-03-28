#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include "matrix.h"
#include "activation.h"
#include <utility>

template <class T>
using pair = std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>>;

template <class T>
using miniBatchVector = std::vector<std::vector<std::vector<Matrix<T>>>>;

template <class T>
using dataVector = std::vector<std::vector<Matrix<T>>>;

template <class T>
using matrixVector = std::vector<Matrix<T>>;

class NeuralNet
{
    /**
     * ``_sizes`` contains numbers of neurons in each layer
     *  first layer is assumed to be input layer (is without bias) 
     *  the last layer - output
    **/
    int _numLayers;
    std::vector<int> _sizes;
    matrixVector<double> _weights;
    matrixVector<double> _biases;

  public:
    NeuralNet(std::vector<int>);
    matrixVector<double> getBiases() { return _biases; }
    matrixVector<double> getWeights() { return _weights; }
    Matrix<double> feedforward(Matrix<double>);
    void feedforward(matrixVector<double> &, Matrix<double> &, matrixVector<double> &);
    double loss(Matrix<double> &, Matrix<double> &);
    void SGD(dataVector<double> &, int, int, double, dataVector<double> &);
    int predict(Matrix<double> &);
    double accuracy(dataVector<double> &);
    void updateMiniBatch(dataVector<double> &, double);
    void updateWeightsAndBiases(matrixVector<double> &, matrixVector<double> &, unsigned int, double);
    pair<double> backprop(Matrix<double> &, Matrix<double> &);
    Matrix<double> costDerivative(Matrix<double>, Matrix<double>);
    Matrix<double> confusionMatrix(dataVector<double>);
    void backwardPass(matrixVector<double> &, matrixVector<double> &,
                      matrixVector<double> &, matrixVector<double> &, Matrix<double> &);
};

dataVector<double> convertData(Matrix<double>, Matrix<double>);
void shuffleData(dataVector<double> &);
miniBatchVector<double> splitIntoMiniBatches(dataVector<double> &, int);
std::pair<dataVector<double>, dataVector<double>> trainTestSplit(dataVector<double> &, double);
matrixVector<double> fillZeros(matrixVector<double> &);

#endif // !__NEURAL_NET__
