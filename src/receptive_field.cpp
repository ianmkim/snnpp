#include <iostream>

#include <vector>
#include <opencv2/opencv.hpp>

#include "params.hpp"
#include "utils.hpp"
#include "receptive_field.hpp"


using namespace std;

/*
 * produce_receptive_field_sse
 * produces a receptive field given a reference to the input image
 * by first padding the image, then using an on-center 5x5 kernel
 * we convolve the image to produce the membrane potential which
 * we write to an already allocated potential array.
 *
 * This function is the SSE optimized version of the other
 * produce_receptive_field_sse functions.
 *
 * THE CALLER IS RESPONSIBLE FOR ALLOCATING ENOUGH SPACE TO HOLD
 * THE POTENTIAL WHICH SHOULD BE THE SAME DIMENSIONS AS THE INPUT IMAGE
 *
 * @param cv mat image
 * @param 2d kernel to convolve the image with
 * @param pre allocated potential 2d array that has same dims as the image
 */
void produce_receptive_field_sse(cv::Mat &inp_unpadded,
                                const vector<vector<float>> &kernel,
                                vector<vector<float>> &potential){
    assert(potential.size() == inp_unpadded.rows && potential.at(0).size() == inp_unpadded.cols);

    // pad the image with 0s so that we don't have to do bounds checking
    cv::Mat inp;
    cv::copyMakeBorder(inp_unpadded, inp, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));

    // the range we need to iterate through to perform convolution with the given kernel
    int ran[5] = {-2, -1, 0, 1, 2};

    // x and y offsets
    int ox = 2;
    int oy = 2;

    // dot_sse uses the m128 four float register
    // so we want 28 numbers so that they can divide nicely into
    // 4 * 7
    // 5 * 5 = 25 for total number of elements in the kernel
    // 3 as padding
    const int size = 5 * 5 + 3;
    float arr1[size];
    float arr2[size];

    // iterate through the entire image membrane potential
    for(int i = 0; i < potential.size(); i++){
        for(int j = 0; j < potential[0].size(); j++){
            int ptr = 0;

            // iterate through the entire kernel
            for(int m_i = 0; m_i < 5; m_i++){
                int m = ran[m_i];
                for(int n_i = 0; n_i < 5; n_i++){
                    int n = ran[n_i];
                    uchar pixel_uc = inp.at<uchar>(i+m+2, j+n+2);

                    // the value for the current pixel can be calculated
                    // as a dot product of the kernel along with the rest of the image
                    // within the 5x5 sliding window
                    // instead of performing the dot product iteratively, we do it with
                    // SIMD intrinsics at the very end
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

/*
 * produce_receptive_field
 * produces a receptive field given a reference to the input image
 * by first padding the image, then using an on-center 5x5 kernel
 * we convolve the image to produce the membrane potential which
 * we write to an already allocated potential array.
 *
 * @param cv mat image
 * @param 2d kernel to convolve the image with
 */
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
