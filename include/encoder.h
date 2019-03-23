#ifndef __ENCODER__H__
#define __ENCODER__H__

#include <string>
#include <vector>
#include <map>
#include "activation.h"

class Encoder
{
    std::map<std::string, int> _classes;
    std::vector<int> _labels;
    int _nClasses;

  public:
    Encoder() = default;
    void fit(std::vector<std::string>);
    std::vector<int> getLabels() { return _labels; }
    std::map<std::string, int> getClasses() { return _classes; }
    int getNClasses() { return _nClasses; }
};

class OneHotEncoder
{
    Matrix<int> _oneHot;
    std::vector<int> _labels;

  public:
    OneHotEncoder() = default;
    void toOneHot(std::vector<int>);
    void fromOneHot(Matrix<int>);
    Matrix<int> getOneHot() { return _oneHot; }
    std::vector<int> getLabels() { return _labels; }
};

void removeDuplicates(std::vector<std::string> &);

#endif