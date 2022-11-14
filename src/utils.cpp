#include "utils.hpp"

#include <math.h>
#include <random>

#include <opencv2/opencv.hpp>

/*
 * sameFloat
 * Equality checks with floats given an epsilon value
 *
 * @param float first number
 * @param float second number
 * @param float error term epsilon
 *
 * @return boolean whether numbers are same or not
 */
bool sameFloat(float x, float y, float epsilon){
    return fabs(x - y) < epsilon;
}

/*
 * genRandom
 * given a low bound and high bound, calculates a random
 * number within the range
 *
 * @param float min number
 * @param float max number
 *
 * @returns generated random number
 */
float genRandom(float low, float high){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> distr(low, high);
    return distr(gen);
}

/*
 * toCVMat
 * converts a 2D vector into an openCV image. 1Channel only
 *
 * @param 2d vector that represents the image
 * @param float multiple to scale each pixel vlaues as
 */
cv::Mat toCVMat(const std::vector<std::vector<float>> &vecIn, const float multiple){
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

/*
 * dot_sse
 * performs the dot product of arr1 and arr2. Optimized with
 * SSE SIMD intrinsics.
 *
 * The float array parameters MUST be heap allocated because they
 * need to extended if they are not a multiple of 4
 *
 * @param float array arr1
 * @param float array arr2
 *
 * @return float dot product of arr1 and arr2
 */
float dot_sse(float* arr1, float* arr2, int len){
    // find out how many elements should be added at the end to pad it
    int num_to_pad = (ceil((double)len/4.0) * 4) - len;
    
    // if we have to pad
    if(num_to_pad > 0){
        // pad both arrays and set the remainder of the elements to 0 so that
        // they won't have any effect on the actual dot product
        size_t new_size = len + num_to_pad;
        float* realloced1 = (float*)realloc(arr1, new_size * sizeof(float));
        if(realloced1){
            arr1 = realloced1;
            memset(arr1+len, 0.0, num_to_pad * sizeof(float));
        } else return -1;

        float* realloced2 = (float*)realloc(arr2, new_size * sizeof(float));
        if(realloced2){
            arr2 = realloced2;
            memset(arr2+len, 0.0, num_to_pad * sizeof(float));
        } else return -1;
    }

    float total;

    int i;
    __m128 num1, num2, num3, num4;
    num4 = _mm_setzero_ps();
    // go through the array in increments of four
    for(i = 0; i < len + num_to_pad; i+=4){
        // perform dot product four elements at a time
        num1 = _mm_loadu_ps(arr1+i);
        num2 = _mm_loadu_ps(arr2+i);
        num3 = _mm_mul_ps(num1, num2);
        num3 = _mm_hadd_ps(num3, num3);
        num4 = _mm_add_ps(num4, num3);
    }

    num4 = _mm_hadd_ps(num4, num4);
    _mm_store_ss(&total, num4);

    return total;
}

/*
 * dot_sse
 * performs the dot product of arr1 and arr2. Optimized with
 * SSE SIMD intrinsics.
 *
 * SIDE EFFECT: the vectors will be resized to a multiple of 4
 * if and the remaining elements will be filled with 0 values
 * if the vector is not a multiple of 4 already.
 *
 * @param float vector v1
 * @param float vector v2
 *
 * @return float dot product of v1 and v2
 */
float dot_sse(std::vector<float> &v1, std::vector<float> &v2){
    assert(v1.size() == v2.size());

    int num_to_pad = (ceil((double)v1.size()/4.0) * 4) - v1.size();
    for(int i = 0; i < num_to_pad; i++){
        v1.push_back(0);
        v2.push_back(0);
    }

    float* arr1 = v1.data();
    float* arr2 = v2.data();

    float arr[4];
    float total;

    int i;
    __m128 num1, num2, num3, num4;
    num4 = _mm_setzero_ps();
    for(i = 0; i < v1.size() + num_to_pad; i+=4){
        num1 = _mm_loadu_ps(arr1+i);
        num2 = _mm_loadu_ps(arr2+i);
        num3 = _mm_mul_ps(num1, num2);
        num3 = _mm_hadd_ps(num3, num3);
        num4 = _mm_add_ps(num4, num3);
    }
    num4 = _mm_hadd_ps(num4, num4);
    _mm_store_ss(&total, num4);

    for(int i = 0; i < num_to_pad; i++){
        v1.pop_back();
        v2.pop_back();
    }

    return total;
}
