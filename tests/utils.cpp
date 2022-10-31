#include "doctest/doctest.h"

#include <vector>

#include "utils.hpp"

using namespace std;

TEST_CASE("argmax"){
    vector<int> v1 = {1,2,3,4,100,2,3};
    CHECK(argmax(v1) == 4);
}

TEST_CASE("Dot product"){
    vector<int> v1 = {1, 2, 3, 4, 5};
    vector<int> v2 = {6, 4, 6, 2, 3};

    int dot_answer = dot<int>(v1, v2);
    CHECK(dot_answer == 55);
}

TEST_CASE("slice col"){
    vector<vector<int>> v1 = {
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12},
    };
    
    vector<int> slice = slice_col<int>(3, v1);
    CHECK(slice.at(0) == 4);
    CHECK(slice.at(1) == 8);
    CHECK(slice.at(2) == 12);
}

TEST_CASE("slice row"){
    vector<vector<int>> v1 = {
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12},
    };

    vector<int> slice = slice_row<int>(2, v1);
    CHECK(slice.at(0) == 9);
    CHECK(slice.at(1) == 10);
    CHECK(slice.at(2) == 11);
    CHECK(slice.at(3) == 12);
}