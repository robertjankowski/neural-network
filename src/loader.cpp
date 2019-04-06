#include "loader.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

Loader::Loader(const char *path, char delimiter)
{
    std::ifstream file(path, std::ios::in);
    std::string line;
    if (file.is_open())
    {
        while (getline(file, line))
        {
            std::vector<std::string> s = splitByDelim(line, delimiter);
            if (!s.empty())
            {
                double sepalL = std::stod(s.at(0));
                double sepalW = std::stod(s.at(1));
                double petalL = std::stod(s.at(2));
                double petalW = std::stod(s.at(3));
                std::string c = s.at(4);
                _iris.push_back(Iris(sepalL, sepalW, petalL, petalW, c));
                _classes.push_back(c);
            }
        }
        file.close();
    }
    else
        throw std::invalid_argument("Unable to open file");
}

Matrix<double> Loader::getInput()
{
    // shape: 150x4
    Matrix<double> features(_iris.size(), _iris[0].getFeatures().cols());
    for (int i = 0; i < features.rows(); ++i)
    {
        features.setOneRow(i, _iris[i].getFeatures());
    }
    return features;
}

std::vector<std::string> splitByDelim(std::string input, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenString(input);
    while (getline(tokenString, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}
