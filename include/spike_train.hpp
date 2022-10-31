#ifndef SPIKE_TRAIN_H
#define SPIKE_TRAIN_H

#include <vector>

#include "params.hpp"

using namespace std;

float scale(float inp, 
        float min_potential, 
        float max_potential, 
        float min_rate,
        float max_rate);


vector<vector<float>> encode(vector<vector<float>> &potential);
#endif