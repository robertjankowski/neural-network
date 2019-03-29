#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "../include/neural_net.h"

TEST_CASE("Testing matrix", "[matrix]")
{
    Matrix<double> A(2, 3);
    A.fill(1);
    Matrix<double> B(3, 2);
    Matrix<double> D(4, 4);
    Matrix<double> A2(2, 3);

    SECTION("Get one rows")
    {
        auto row = A.getOneRow(0);
        REQUIRE(row == std::vector<double>{1, 1, 1});
    }

    SECTION("Matrix transpose")
    {
        auto At = A.transpose();
        REQUIRE(At.cols() == 2);
        REQUIRE(At.rows() == 3);
    }

    SECTION("Matrix multiplication")
    {
        auto C = mul(A, B);
        REQUIRE(C.rows() == 2);
        REQUIRE(C.cols() == 2);
    }

    SECTION("Exception matrix multiplication")
    {
        REQUIRE_THROWS_AS(mul(A, D), std::invalid_argument);
    }

    SECTION("Hadamard product")
    {
        REQUIRE_NOTHROW(dot(A, A2));
        auto AA = dot(A, A2);
        REQUIRE(AA.cols() == A.cols());
        REQUIRE(AA.rows() == A.rows());
    }

    SECTION("Exception hadamard product")
    {
        REQUIRE_THROWS_AS(dot(A, B), std::invalid_argument);
    }
}

TEST_CASE("Testing neural network", "[neural net]")
{
    Matrix<double> input(1, 4);
    input.fillGauss(0, 1);
    Matrix<double> labels(1, 3);
    for (int i = 0; i < labels.cols(); ++i)
        labels.set(0, i, i % 2);

    std::vector<int> sizes = {4, 8, 3};
    NeuralNet nn(sizes);

    SECTION("Weights and biases shapes")
    {
        auto weights = nn.getWeights();
        auto w1 = weights.at(0);
        REQUIRE(w1.rows() == 8);
        REQUIRE(w1.cols() == 4);

        auto w2 = weights.at(1);
        REQUIRE(w2.rows() == 3);
        REQUIRE(w2.cols() == 8);

        auto biases = nn.getBiases();
        auto b1 = biases.at(0);
        REQUIRE(b1.rows() == 8);
        REQUIRE(b1.cols() == 1);

        auto b2 = biases.at(1);
        REQUIRE(b2.rows() == 3);
        REQUIRE(b2.cols() == 1);
    }
}