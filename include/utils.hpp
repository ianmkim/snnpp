#ifndef UTILS_H
#define UTILS_H
    
#include "sse2neon.h"

#include <vector>
#include <opencv2/opencv.hpp>

bool sameFloat(float x, float y, float epsilon=0.001);

float genRandom(float low=0.0, float high=1.0);

cv::Mat toCVMat(const std::vector<std::vector<float>> &vecIn, const float multiple);

float dot_sse(std::vector<float> &v1, std::vector<float> &v2);


/*
 * from https://cplusplus.com/forum/general/216928/
 */
template<typename T> T interpolate( std::vector<T> &xData, std::vector<T> &yData, T x, bool extrapolate=false){
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
    std::vector<T> col;
    for(int i = 0; i < v.size(); i++){
        col.push_back(v.at(i).at(idx));
    } return col;
}

template <typename T>
std::vector<T> slice_row(int idx, std::vector<std::vector<T>> &v){
    assert(idx < v.size());
    std::vector<T> row;
    for(T item : v.at(idx)){
        row.push_back(item);
    } return row;
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