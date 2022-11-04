#include "utils.hpp"

#include <math.h>
#include <random>

#include <opencv2/opencv.hpp>

bool sameFloat(float x, float y, float epsilon){
    return fabs(x - y) < epsilon;
}

float genRandom(float low, float high){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> distr(low, high);
    return distr(gen);
}


cv::Mat toCVMat(const std::vector<std::vector<float>> vecIn, const float multiple){
    cv::Mat matOut(vecIn.size(), vecIn.at(0).size(), CV_8UC1);
    for (int i = 0; i < matOut.rows; ++i) {
        for (int j = 0; j < matOut.cols; ++j) {
            float pixel_float = fmax(fmin(multiple * vecIn.at(i).at(j), 255), 0);
            uchar pixel = (uchar) pixel_float;
            matOut.at<uchar>(i, j) =pixel;
        }
    }
    return matOut;
}