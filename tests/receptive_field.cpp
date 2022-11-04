#include "doctest/doctest.h"
#include <iostream>
#include <opencv2/opencv.hpp>

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