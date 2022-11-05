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

    vector<vector<float>> pot = produce_receptive_field(image);
    
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
TEST_CASE("Receptive Field Convolution METAL"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg",
        //"large_img.jpg",
        IMREAD_GRAYSCALE);

    CHECK(image.empty() == false);

    vector<vector<float>>pot_metal(
        image.rows, 
        vector<float>(image.cols, 0.0));


    produce_receptive_field_metal(image, pot_metal);
    vector<vector<float>> pot = produce_receptive_field(image);

    for(int i = 0; i < pot.size(); i++){
        for(int j = 0; j < pot.at(i).size(); j++){
            CHECK(sameFloat(pot_metal[i][j], pot[i][j]));
        }
    }
}

TEST_CASE("Receptive Field Convolution Benchmark"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg",
        //"large_img.jpg",
        IMREAD_GRAYSCALE);

    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++){
        vector<vector<float>> pot = produce_receptive_field(image);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time spent doing CPU receptive field convolution: " << duration.count() << "ms" << endl;

    vector<vector<float>> potential(
        image.rows, 
        vector<float>(image.cols, 0.0));


    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++){
        produce_receptive_field_metal(image, potential);
    }
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Time spent doing GPU Metal receptive field convolution: " << duration.count() << "ms" << endl;
}
#endif