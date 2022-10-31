#include "doctest/doctest.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include "spike_train.hpp"
#include "receptive_field.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;



TEST_CASE("Testing scaling function"){
    float scaled;

    scaled = scale(0, -1.069, 2.781, 1, 20);
    //std::cout << "scaled value: " << scaled << " | gt: " << 6.2755844 << std::endl;
    CHECK(sameFloat(scaled, 6.2755844));

    scaled = scale(1, -1.069, 2.781, 1, 20);
    //std::cout << "scaled value: " << scaled << " | gt: " << 11.210649 << std::endl;
    CHECK(sameFloat(scaled, 11.210649));

    scaled = scale(2, -1.069, 2.781, 1, 20);
    //std::cout << "scaled value: " << scaled << " | gt: " << 16.145714 << std::endl;
    CHECK(sameFloat(scaled, 16.145714));

    scaled = scale(-1, -1.069, 2.781, 1, 20);
    //std::cout << "scaled value: " << scaled << " | gt: " << 1.340519 << std::endl;
    CHECK(sameFloat(scaled, 1.340519));

    scaled = scale(2.5, -1.069, 2.781, 1, 20);
    //std::cout << "scaled value: " << scaled << " | gt: " << 18.613246 << std::endl;
    CHECK(sameFloat(scaled, 18.613246));
}

TEST_CASE("Testing Spike Train"){
    Mat image = imread(
        "mnist_set/0/img_1.jpg", 0);

    CHECK(image.empty() == false);
    vector<vector<float>> pot = produce_receptive_field(image);
    
    CHECK(pot.size() >= 1);
    CHECK(pot[0].size() >= 1);

    vector<vector<float>> train = encode(pot);
    CHECK(train.size() == 784);
    CHECK(train.at(0).size() == 201);

    ifstream gt_file("train_groundtruth.txt");
    string line;
    int train_indx = 0;
    int num_wrong = 0;
    while(getline(gt_file, line)){
        for(int i = 0; i < line.size(); i++){
            char num_str[2] = {line.c_str()[i], '\0'};
            if(stoi(num_str) != ceil(train.at(i).at(train_indx))){
                num_wrong++;
            }
        } train_indx++;
    }

    // if we got more than 0.1% error from the python ground truth
    CHECK(num_wrong <= (int)((784*201) * 0.001));
}