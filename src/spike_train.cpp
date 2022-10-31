#include <iostream>
#include <vector>
#include <stdexcept>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "params.hpp"

using namespace std;
using namespace cv;

float scale(float inp, 
            float min_potential, 
            float max_potential, 
            float min_rate,
            float max_rate){

    return ((max_rate - min_rate) * ((inp - min_potential) / (max_potential - min_potential))) + min_rate;
}

vector<vector<float>> encode(vector<vector<float>> &potential){
    vector<vector<float>> train;

    for(int i = 0; i < Params::pixel_x; i++){
        for(int j = 0; j < Params::pixel_x; j++){
            vector<float> temp(Params::time+1, 0.0);
            
            float freq = scale(
                    potential.at(i).at(j),
                    -1.069, 2.781, 
                    1, 20);

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
