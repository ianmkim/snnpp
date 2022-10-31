#include "utils.hpp"
#include <math.h>
#include <random>


bool sameFloat(float x, float y, float epsilon){
    return fabs(x - y) < epsilon;
}

float genRandom(float low, float high){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> distr(low, high);
    return distr(gen);
}

