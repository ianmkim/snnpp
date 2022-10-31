#ifndef UTILS_H
#define UTILS_H

#include <vector>

bool sameFloat(float x, float y, float epsilon=0.001);

float genRandom(float low=0.0, float high=1.0);


template <typename T>
T dot(std::vector<T> &v1, std::vector<T> &v2){
    assert(v1.size() == v2.size());
    assert(v1.size() > 0);
    T sum = v1[0] * v2[0];
    for(int i = 1; i < v1.size(); i++)
        sum += v1[i] * v2[i];
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