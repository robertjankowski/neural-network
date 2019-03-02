#ifndef __MATRIX__H__
#define __MATRIX__H__

#include <vector>
#include <stdexcept>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <cstdlib>
#include <ctime>

template <class T>
class Matrix
{
    static_assert(std::is_arithmetic<T>::value, "Matrix must be numeric");

    std::vector<std::vector<T>> _X;

  public:
    Matrix() = default;
    Matrix(const int &, const int &);
    Matrix(const Matrix &) = default;
    const T &at(int, int);
    void set(int, int, const T &);
    int rows() { return _X.size(); }
    int cols() { return _X[0].size(); }
    void resize(const int &, const int &);
    typename std::vector<std::vector<T>>::iterator begin() { return _X.begin(); }
    typename std::vector<std::vector<T>>::iterator end() { return _X.end(); }
    Matrix operator*(double);
    Matrix operator-(Matrix &);
    Matrix operator+(Matrix &);
    void fillGauss(double, double);
    void fillUniform(double, double);
    void fillRandom(int);
    void fill(T);
    std::tuple<int, int> getShape();
    void showShape();
    Matrix transpose();
    void setOneRow(int, std::vector<T>);
    std::vector<T> getOneRow(int);
    Matrix<double> convertToDouble();
    Matrix<T> shuffleRows();
    Matrix<T> concatMatrix(Matrix<T> &);
    std::vector<Matrix<T>> splitMatrix(int);

    template <class U>
    friend std::ostream &operator<<(std::ostream &, Matrix<U>);
    template <class U>
    friend Matrix<U> mul(Matrix<U> &, Matrix<U> &);
    template <class U>
    friend std::vector<std::vector<Matrix<U>>> trainTestSplit(std::vector<Matrix<U>>, double);
};

template <class T>
Matrix<T>::Matrix(const int &rows, const int &cols)
{
    Matrix<T>::resize(rows, cols);
}

template <class T>
const T &Matrix<T>::at(int i, int j)
{
    return _X[i][j];
}

template <class T>
void Matrix<T>::set(int i, int j, const T &newVal)
{
    _X[i][j] = newVal;
}

template <class T>
void Matrix<T>::resize(const int &rows, const int &cols)
{
    _X.resize(rows, std::vector<T>(cols));
}

template <class T>
Matrix<T> Matrix<T>::operator*(double scalar)
{
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            T oldVal = at(i, j);
            set(i, j, oldVal * scalar);
        }
    }
    return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator+(Matrix &other)
{
    if (cols() != other.cols() || rows() != other.rows())
    {
        throw std::invalid_argument("Cannot add matrices with different shape");
    }
    Matrix<T> result(rows(), cols());
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            result.set(i, j, at(i, j) + other.at(i, j));
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::operator-(Matrix &other)
{

    if (cols() != other.cols() || rows() != other.rows())
    {
        throw std::invalid_argument("Cannot substract matrices with different shape");
    }
    Matrix<T> result(rows(), cols());
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            result.set(i, j, at(i, j) - other.at(i, j));
        }
    }
    return result;
}

template <class T>
void Matrix<T>::fillGauss(double mean, double stdiv)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<> distr(mean, stdiv);
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            T val = distr(generator);
            set(i, j, val);
        }
    }
}

template <class T>
void Matrix<T>::fillUniform(double from, double to)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distr(from, to);
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            T val = distr(generator);
            set(i, j, val);
        }
    }
}

template <class T>
void Matrix<T>::fillRandom(int max)
{
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            set(i, j, rand() % max);
        }
    }
}

template <class T>
void Matrix<T>::fill(T num)
{
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            set(i, j, num);
        }
    }
}

template <class T>
std::tuple<int, int> Matrix<T>::getShape()
{
    return std::make_tuple(rows(), cols());
}

template <class T>
void Matrix<T>::showShape()
{
    std::tuple<int, int> shape = getShape();
    std::cout << "(" << std::get<0>(shape) << ","
              << std::get<1>(shape) << ")" << std::endl;
}

template <class T>
Matrix<T> Matrix<T>::transpose()
{
    Matrix<T> transposed(cols(), rows());
    for (int i = 0; i < cols(); ++i)
    {
        for (int j = 0; j < rows(); ++j)
        {
            transposed.set(i, j, at(j, i));
        }
    }
    return transposed;
}

template <class T>
void Matrix<T>::setOneRow(int row, std::vector<T> input)
{
    if (input.size() != (unsigned)cols())
    {
        throw std::invalid_argument("Input vector size need to be the same as col()");
    }
    for (int i = 0; i < cols(); ++i)
    {
        set(row, i, input.at(i));
    }
}

template <class T>
std::vector<T> Matrix<T>::getOneRow(int pos)
{
    std::vector<T> oneRow;
    oneRow.reserve(cols());
    for (int i = 0; i < cols(); ++i)
    {
        oneRow.push_back(at(pos, i));
    }
    return oneRow;
}

template <class T>
Matrix<double> Matrix<T>::convertToDouble()
{
    Matrix<double> converted(rows(), cols());
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            double val = at(i, j);
            converted.set(i, j, val);
        }
    }
    return converted;
}

template <class T>
Matrix<T> Matrix<T>::shuffleRows()
{
    std::srand(unsigned(std::time(0)));
    std::random_shuffle(_X.begin(), _X.end());
    return *this;
}

template <class T>
Matrix<T> Matrix<T>::concatMatrix(Matrix<T> &Y)
{
    if (rows() != Y.rows())
    {
        throw std::invalid_argument("Cannot concat matrices due to wrong rows size");
    }
    int newCols = cols() + Y.cols();
    Matrix<T> out(rows(), newCols);
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < newCols; ++j)
        {
            if (j < cols())
            {
                T val = at(i, j);
                out.set(i, j, val);
            }
            else
            {
                T val = Y.at(i, j - cols());
                out.set(i, j, val);
            }
        }
    }
    return out;
}

template <class T>
std::vector<Matrix<T>> Matrix<T>::splitMatrix(int pos)
{
    Matrix<T> left(rows(), pos);
    Matrix<T> right(rows(), cols() - pos);
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            if (j < pos)
            {
                T val = at(i, j);
                left.set(i, j, val);
            }
            else
            {
                T val = at(i, j);
                right.set(i, j - pos, val);
            }
        }
    }
    std::vector<Matrix<T>> out{left, right};
    return out;
}

template <class U>
std::ostream &operator<<(std::ostream &stream, Matrix<U> m)
{
    for (const auto vec : m)
    {
        for (const auto v : vec)
        {
            stream << v << " ";
        }
        stream << "\n";
    }
    return stream;
}

template <class U>
Matrix<U> mul(Matrix<U> &A, Matrix<U> &B)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument("Cannot multiply matrices due to wrong shape");
    }
    Matrix<U> res(A.rows(), B.cols());
    U accumulator;
    for (int i = 0; i < A.rows(); ++i)
    {
        for (int j = 0; j < B.cols(); ++j)
        {
            accumulator = 0.0;
            for (int k = 0; k < B.rows(); ++k)
            {
                accumulator += A.at(i, k) * B.at(k, j);
            }
            res.set(i, j, accumulator);
        }
    }
    return res;
}

template <class U>
std::vector<std::vector<Matrix<U>>> trainTestSplit(std::vector<Matrix<U>> X, double testSize)
{

    // concat matrices, shuffle and split
    Matrix<U> features = X.at(0);
    Matrix<U> labels = X.at(1);
    Matrix<U> all = features.concatMatrix(labels);
    Matrix<U> allRandom = all.shuffleRows();

    int testRows = int(features.rows() * testSize);
    int trainRows = features.rows() - testRows;

    Matrix<U> test(testRows, allRandom.cols());
    Matrix<U> train(trainRows, allRandom.cols());

    for (int i = 0; i < allRandom.rows(); ++i)
    {
        if (i < testRows)
        {
            test.setOneRow(i, allRandom.getOneRow(i));
        }
        else
        {
            train.setOneRow(i - testRows, allRandom.getOneRow(i));
        }
    }
    std::vector<Matrix<U>> testVec = test.splitMatrix(features.cols());
    std::vector<Matrix<U>> trainVec = train.splitMatrix(features.cols());
    std::vector<std::vector<Matrix<U>>> trainTestVec = {testVec, trainVec};
    return trainTestVec;
}

#endif // __MATRIX__H__