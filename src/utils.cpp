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

float dot_sse(float* arr1, float* arr2, int len){
    int num_to_pad = len % 4;
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

    float arr[4]; 
    float total; 
    
    int i;
    __m128 num1, num2, num3, num4;
    num4 = _mm_setzero_ps();
    for(i = 0; i < len + num_to_pad; i+=4){
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

float dot_sse(std::vector<float> &v1, std::vector<float> &v2){
    assert(v1.size() == v2.size());

    int num_to_pad = v1.size() % 4;
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