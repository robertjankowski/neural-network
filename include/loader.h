#ifndef __LOADER__H__
#define __LOADER__H__

#include <string>
#include <vector>
#include "dataset.h"
#include "matrix.h"

class Loader
{
    std::vector<Iris> _iris;
    std::vector<std::string> _classes;

  public:
    Loader(const char *, char);
    std::vector<Iris> getDataset() { return _iris; }
    std::vector<std::string> getClasses() { return _classes; }
    Matrix<double> getInput();
};

std::vector<std::string> splitByDelim(std::string, char);

#endif