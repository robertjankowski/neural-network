#ifndef __ACTIVATION__H__
#define __ACTIVATION__H__

#include <vector>
#include <functional>
#include "matrix.h"

namespace Activation
{
double sigmoid(double);
double tanh(double);
double relu(double);
double leakyRelu(double);
double sigmoidDerivative(double);
} // namespace Activation

Matrix<double> applyActivation(std::function<double(double)>, Matrix<double>);

#endif // __ACTIVATION__H__