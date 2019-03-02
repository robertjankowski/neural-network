#ifndef __DATASET__H__
#define __DATASET__H__

#include <string>
#include <iostream>
#include <vector>

class Iris
{
    double _sepalLength;
    double _sepalWidth;
    double _petalLength;
    double _petalWidth;
    std::string _class;

  public:
    Iris(double, double, double, double, std::string);
    friend std::ostream &operator<<(std::ostream &os, const Iris &i);
    std::vector<double> getFeatures();
};

std::ostream &operator<<(std::ostream &os, const Iris &i);

#endif