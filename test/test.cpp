#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE("first test", "[test]")
{
    REQUIRE(1 == 1);
}
