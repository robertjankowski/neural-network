#include "dataset.h"
#include <string>
#include <iostream>

Iris::Iris(double sL, double sW, double pL, double pW, std::string c)
    : _sepalLength(sL), _sepalWidth(sW), _petalLength(pL),
      _petalWidth(pW), _class(c) {}

std::vector<double> Iris::getFeatures()
{
    std::vector<double> vec{_sepalLength, _sepalWidth, _petalLength, _petalWidth};
    return vec;
}

std::ostream &operator<<(std::ostream &os, const Iris &i)
{
    os << "Sepal Width: " << i._sepalWidth << "\tSepal Length"
       << i._sepalLength << "\tPetal Width: " << i._petalWidth
       << "\tPetal Length: " << i._petalLength << "\tClass: "
       << i._class;
    return os;
}