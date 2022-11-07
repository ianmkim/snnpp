#include "doctest/doctest.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "receptive_field.hpp"
#include "utils.hpp"

using namespace cv;


TEST_CASE("Receptive field convolution"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg",
        IMREAD_GRAYSCALE);

    CHECK(image.empty() == false);

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3},
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };

    vector<vector<float>> pot = produce_receptive_field(image, kernel);
    
    CHECK(pot.size() >= 1);
    CHECK(pot[0].size() >= 1);

    float min = pot[0][0];
    float max = pot[0][0];
    
    for(vector<float> row : pot){
        float min_row = *min_element(row.begin(), row.end());
        float max_row = *max_element(row.begin(), row.end());

        if(min > min_row) min = min_row;
        if(max < max_row) max = max_row;
    }

    std::cout << min << std::endl;
    std::cout << max << std::endl;

    Mat img = toCVMat(pot, (float)255.0);
    imwrite("test_results/pot_1.jpg", img);

    CHECK(sameFloat(min, -1.1122549019));
    CHECK(sameFloat(max, 3.02107843137));
}

#ifdef __APPLE__
TEST_CASE("Receptive Field Convolution SSE"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg",
        //"large_img.jpg",
        IMREAD_GRAYSCALE);

    CHECK(image.empty() == false);

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3},
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };

    vector<vector<float>> pot_sse(
        image.rows,
        vector<float>(image.cols, 0.0));

    produce_receptive_field_sse(image, kernel, pot_sse);
    vector<vector<float>> pot = produce_receptive_field(image, kernel);
    for(int i = 0; i < pot.size(); i++){
        for(int j = 0; j < pot.at(i).size(); j++){
            CHECK(sameFloat(pot_sse[i][j], pot[i][j]));
        }
    }
}

TEST_CASE("Receptive Field Convolution Benchmark"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg",
        //"large_img.jpg",
        IMREAD_GRAYSCALE);

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3},
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };


    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < 50; i++){
        vector<vector<float>> pot = produce_receptive_field(image, kernel);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time spent doing CPU receptive field convolution: " << duration.count() << "ms" << endl;

    vector<vector<float>> potential(
        image.rows, 
        vector<float>(image.cols, 0.0));

    vector<vector<float>> pot_sse(
        image.rows,
        vector<float>(image.cols, 0.0));
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < 50; i++){
        produce_receptive_field_sse(image, kernel, pot_sse);
    }
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Time spent doing SSE receptive field convolution: " << duration.count() << "ms" << endl;
}
#endif