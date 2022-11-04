#ifndef RECEPTIVE_FIELD_H
#define RECEPTIVE_FIELD_H
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

vector<vector<float>> produce_receptive_field(cv::Mat inp);

#endif