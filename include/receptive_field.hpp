#ifndef RECEPTIVE_FIELD_H
#define RECEPTIVE_FIELD_H
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

void produce_receptive_field_sse(cv::Mat &inp, 
                                const vector<vector<float>> &kernel, 
                                vector<vector<float>> &potential);

vector<vector<float>> produce_receptive_field(cv::Mat &inp, vector<vector<float>> &kernel);

#ifdef __APPLE__


//vector<vector<float>> produce_receptive_field_metal(cv::Mat &inp);
/*
void prepare_metal_device(MTL::Device *device, MetalAdder* adder);
void produce_receptive_field_metal(cv::Mat &inp, vector<vector<float>> &potential, MTL::Device *device, MetalAdder* adder);
*/

#endif 

#endif