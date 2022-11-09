#ifndef SPIKE_TRAIN_H
#define SPIKE_TRAIN_H

#include <vector>

#include "params.hpp"

using namespace std;


/*
 * scale
 * scales an input value between the range up to the range of the
 * output range.
 * 
 * @param float input
 * @param float minimum of range1
 * @param float maxmimum of range 1
 * @param float minimum of range 2
 * @param float maximum of range 2
 * 
 * @return scaled input value
 */
float scale(float inp, 
        float min_potential, 
        float max_potential, 
        float min_rate,
        float max_rate);


/*
 * encode
 * given a 2d membrane potential, this function encodes it into
 * a spike train that can be fed directly into the SNN as input
 */
vector<vector<float>> encode(vector<vector<float>> &potential);
#endif