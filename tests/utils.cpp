#include "doctest/doctest.h"

#include <vector>
#include <random>
#include <chrono>

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

TEST_CASE("Dot product SSE"){
    vector<float> v1 = {1, 2, 3, 4, 5};
    vector<float> v2 = {6, 4, 6, 2, 3};

    float dot_answer = dot_sse(v1, v2);
    CHECK(dot_answer == 55);
}

TEST_CASE("Dot product SSE array"){
    vector<float> vv1 = {1, 2, 3, 4, 5};
    vector<float> vv2 = {6, 4, 6, 2, 3};

    float* v1 = (float*)malloc(vv1.size() * sizeof(float));
    memcpy(v1, vv1.data(), vv1.size() * sizeof(float));
    float* v2 = (float*)malloc(vv2.size() * sizeof(float));
    memcpy(v2, vv2.data(), vv2.size() * sizeof(float));

    float dot_answer = dot_sse(v1, v2, vv1.size());

    free(v1);
    free(v2);

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

TEST_CASE("Benchmark"){
    vector<float> v1;
    vector<float> v2;

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);

    for(int i = 0; i < 1000; i++){
        v1.push_back(dist(e2));
        v2.push_back(dist(e2));
    }


    auto start = chrono::high_resolution_clock::now();
    for(int repeat = 0; repeat < 10000; repeat++){
        int dot_p = dot<float>(v1, v2);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Time spent doing naive dot product: " << duration.count() << "ms" << endl;


    start = chrono::high_resolution_clock::now();
     for(int repeat = 0; repeat <10000; repeat++){
        float dot_p = dot_sse(v1, v2);
    }
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Time spent doing SSE dot product: " << duration.count() << "ms" << endl;
}