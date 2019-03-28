#include "encoder.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>

void Encoder::fit(std::vector<std::string> input)
{
    std::vector<std::string> classes = input;
    removeDuplicates(classes);
    for (unsigned int i = 0; i < classes.size(); ++i)
    {
        _classes.insert(std::pair<std::string, int>(classes.at(i), i));
    }
    for (auto &s : input)
    {
        _labels.push_back(_classes.at(s));
    }
}

void removeDuplicates(std::vector<std::string> &vec)
{
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}

void OneHotEncoder::toOneHot(std::vector<int> input)
{
    int max = *std::max_element(input.begin(), input.end()) + 1;
    int size = input.size();
    // e.g max = 2 -> size must be 3 [1 0 0]
    _oneHot.resize(size, max);
    for (int i = 0; i < size; ++i)
    {
        std::vector<int> row(max);
        for (int j = 0; j < max; ++j)
        {
            _oneHot.set(i, j, 0);
            _oneHot.set(i, input.at(i), 1);
        }
    }
}

void OneHotEncoder::fromOneHot(Matrix<int> input)
{
    int cols = input.cols();
    int rows = input.rows();
    _labels.resize(cols);
    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            if (input.at(i, j) == 1)
                _labels.at(i) = j + 1;
        }
    }
}