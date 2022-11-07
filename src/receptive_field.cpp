#include <iostream>

#include <vector>
#include <opencv2/opencv.hpp>

#include "params.hpp"
#include "utils.hpp"
#include "receptive_field.hpp"


using namespace std;

void produce_receptive_field_sse(cv::Mat &inp_unpadded,
                                const vector<vector<float>> &kernel,
                                vector<vector<float>> &potential){
    assert(potential.size() == inp_unpadded.rows && potential.at(0).size() == inp_unpadded.cols);
    cv::Mat inp;
    cv::copyMakeBorder(inp_unpadded, inp, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));

    int ran[5] = {-2, -1, 0, 1, 2};
    int ox = 2;
    int oy = 2;

    // dot_sse uses the m128 four float register
    // so we want 28 numbers so that they can divide nicely into
    // 4 * 7
    const int size = 5 * 5 + 3;
    float arr1[size];
    float arr2[size];

    for(int i = 0; i < potential.size(); i++){
        for(int j = 0; j < potential[0].size(); j++){
            int ptr = 0;

            for(int m_i = 0; m_i < 5; m_i++){
                int m = ran[m_i];
                for(int n_i = 0; n_i < 5; n_i++){
                    int n = ran[n_i];
                    uchar pixel_uc = inp.at<uchar>(i+m+2, j+n+2);
                    arr1[ptr] = (float)pixel_uc / 255.0;
                    arr2[ptr] = kernel[ox+m][oy+n]; 
                    ptr++;
                }
            }

            float sum = dot_sse(arr1, arr2, size);

            potential[i][j] = sum;
        }
    }
}

vector<vector<float>> produce_receptive_field(cv::Mat &inp, vector<vector<float>> &kernel){
    vector<vector<float>> potential(
        inp.rows,
        vector<float>(inp.cols, 0.0));

    vector<int> ran {-2, -1, 0, 1, 2};
    int ox = 2;
    int oy = 2;

    for(int i = 0; i < potential.size(); i++){
        for(int j = 0; j < potential[0].size(); j++){
            float sum = 0;

            for(int m : ran){
                for(int n : ran){
                    if((i+m) >= 0 &&
                        (i+m) <= Params::pixel_x-1 &&
                        (j+n) >= 0 &&
                        (j+n) <= Params::pixel_x-1){
                        uchar pixel_uc = inp.at<uchar>(i+m, j+n);
                        float pixel = (float)pixel_uc;
                        pixel /= 255.0;
                        sum += kernel.at(ox+m).at(oy+n) *
                                pixel;
                    }
                }
            }

            potential[i][j] = sum;
        }
    }

    /*
    cv::Mat image(potential.size(), potential.at(0).size(), CV_64FC1);
    for(int i = 0; i<image.rows; ++i){
        for(int j = 0; j <image.cols;j++){
            image.at<double>(i, j) = potential.at(i).at(j) * 255.0;
        }
    }
    imwrite("something.png", image);
    */


    return potential;
}
