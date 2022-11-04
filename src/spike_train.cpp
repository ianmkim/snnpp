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

vector<vector<float>> encode(vector<vector<float>> &potential){
    vector<vector<float>> train;

    for(int i = 0; i < Params::pixel_x; i++){
        for(int j = 0; j < Params::pixel_x; j++){
            vector<float> temp(Params::time+1, 0.0);
            
            vector<float> r1 = {-1.069, 2.781};
            vector<float> r2 = {1.0,     20.0};
            float freq = interpolate(r1, r2, potential.at(i).at(j));

            /*
            float freq = scale(
                    potential.at(i).at(j),
                    -1.069, 2.781, 
                    1.0, 20.0);
            */

            if(freq <= 0)
                std::cerr << "Frequency is out of range" << std::endl;

            float freq1 = ceil(600/freq);
            float k = freq1;
            if (potential.at(i).at(j) > 0){
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
