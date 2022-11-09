#ifndef UTILS_H
#define UTILS_H
    
#include "sse2neon.h"

#include <vector>
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
bool sameFloat(float x, float y, float epsilon=0.001);


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
float genRandom(float low=0.0, float high=1.0);

/*
 * toCVMat
 * converts a 2D vector into an openCV image. 1Channel only
 * 
 * @param 2d vector that represents the image
 * @param float multiple to scale each pixel vlaues as
 */
cv::Mat toCVMat(const std::vector<std::vector<float>> &vecIn, const float multiple);

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
float dot_sse(float* arr1, float* arr2, int len);

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
float dot_sse(std::vector<float> &v1, std::vector<float> &v2);


/*
 * from https://cplusplus.com/forum/general/216928/
 */
template<typename T> 
T interpolate( std::vector<T> &xData, std::vector<T> &yData, T x, bool extrapolate=false){
    int size = xData.size();

    // find left end of interval for interpolation
    int i = 0;
    
    // special case: beyond right end
    if ( x >= xData[size - 2] ) {
        i = size - 2;
    } else

    {
        while ( x > xData[i+1] ) i++;
    }

    // points on either side (unless beyond ends)
    T xL = xData[i], yL = yData[i], xR = xData[i+1], yR = yData[i+1];   
    // if beyond ends of array and not extrapolatin
    if ( !extrapolate ){
        if ( x < xL ) yR = yL;
        if ( x > xR ) yL = yR;
    }

    // gradient
    T dydx = ( yR - yL ) / ( xR - xL );                                    
    // linear interpolation
    return yL + dydx * ( x - xL );                                              
}


template <typename T>
T dot(std::vector<T> &v1, std::vector<T> &v2){
    assert(v1.size() == v2.size());
    assert(v1.size() > 0);
    T sum = v1[0] * v2[0];
    for(int i = 1; i < v1.size(); i++)
        sum = sum + v1[i] * v2[i];
    return sum;
}

template <typename T>
std::vector<T> slice_col(int idx, std::vector<std::vector<T>> &v){
    assert(v.size() > 0);
    assert(idx < v.at(0).size());
    std::vector<T> col(v.size());
    int ptr = 0;
    for(int i = 0; i < v.size(); i++){
        col[ptr++] = v.at(i).at(idx);
    } return col;
}

template <typename T>
T* slice_col(int idx, T** v, int width, int height){
    T* out = (T*)malloc(height * sizeof(T));
    if(out == NULL) return NULL;
    int ptr = 0;
    for(int i = 0; i < height; i++){
        out[ptr++] = v[i][idx];
    } return out;
}

template <typename T>
std::vector<T> slice_row(int idx, std::vector<std::vector<T>> &v){
    assert(idx < v.size());
    std::vector<T> row(v.at(idx).size());
    memcpy(row.data(), v.at(idx).data(), v.at(idx).size() * sizeof(T));
    return row;
}

template<typename T>
int argmax(std::vector<T> &inp){
    if(inp.size() == 0) return -1;
    int max = inp.at(0);
    int max_idx = 0;
    for(int i = 0; i < inp.size(); i++){
        if(max < inp.at(i)){
            max = inp.at(i);
            max_idx= i;
        }
    }

    return max_idx;
}

#endif