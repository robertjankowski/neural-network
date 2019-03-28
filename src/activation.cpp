#include "activation.h"
#include <cmath>

Matrix<double> applyActivation(fActivation f, Matrix<double> X)
{
    int cols = X.cols();
    int rows = X.rows();
    Matrix<double> output(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output.set(i, j, f(X.at(i, j)));
        }
    }
    return output;
}

double Activation::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
double Activation::relu(double x)
{
    return fmax(0, x);
}
double Activation::leakyRelu(double x)
{
    if (x < 0)
        return -0.01 * x;
    else
        return x;
}

double Activation::sigmoidDerivative(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}