#include <iostream>
#include <vector>
#include <stdexcept>
#include <math.h>

#include <opencv2/opencv.hpp>
#include "spike_train.hpp"

#include "params.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

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
            float max_rate){
    float old_range = (max_potential - min_potential);
    float new_range = (max_rate - min_rate);
    float new_value = (((inp - min_potential) * new_range) / old_range) + min_rate;
    return new_value;
}


/*
 * encode
 * given a 2d membrane potential, this function encodes it into
 * a spike train that can be fed directly into the SNN as input
 */
vector<vector<float>> encode(vector<vector<float>> &potential){
    // dims will be such that train.size() = input dimensions and
    // train[i].size() = max simulation time
    vector<vector<float>> train;

    // iterate through all the pixels
    for(int i = 0; i < Params::pixel_x; i++){
        for(int j = 0; j < Params::pixel_x; j++){
            vector<float> temp(Params::time+1, 0.0);

            // convert the membrane potential values into
            // spikes range that can be fed into the network
            // constants from the reference implementation
            // frequency will range from 1hz to 20hz in the final spike
            // train since the max_time is 200.
            // so the maximum amount of time we can spike in sequence
            // is every single time unit, which will be 20hz
            vector<float> r1 = {-1.069, 2.781};
            vector<float> r2 = {1.0,     20.0};
            float freq = interpolate(r1, r2, potential.at(i).at(j));

            if(freq <= 0)
                std::cerr << "Frequency is out of range" << std::endl;

            // find the frequency of the spikes generated from this
            // particular part of the membrane potential
            float freq1 = ceil(600/freq);
            float k = freq1;
            if (potential.at(i).at(j) > 0){
                // add regular spikes based on the frequency
                while( k < Params::time + 1 ){
                    temp[k] = 1.0;
                    k += freq1;
                }
            }
            train.push_back(temp);
        }
    }

    return train;
}
