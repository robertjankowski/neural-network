#ifndef __ACTIVATION__H__
#define __ACTIVATION__H__

#include <vector>
#include <functional>
#include "matrix.h"

using fActivation = std::function<double(double)>;

namespace Activation
{
double sigmoid(double);
double relu(double);
double leakyRelu(double);
double sigmoidDerivative(double);
} // namespace Activation

Matrix<double> applyActivation(fActivation, Matrix<double>);

#endif // __ACTIVATION__H__