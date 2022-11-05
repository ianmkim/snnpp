#ifndef RECEPTIVE_FIELD_H
#define RECEPTIVE_FIELD_H
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

vector<vector<float>> produce_receptive_field(cv::Mat &inp);

#ifdef __APPLE__

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "metal_adder.hpp"

//vector<vector<float>> produce_receptive_field_metal(cv::Mat &inp);

void produce_receptive_field_metal(cv::Mat &inp, vector<vector<float>> &potential);
/*
void prepare_metal_device(MTL::Device *device, MetalAdder* adder);
void produce_receptive_field_metal(cv::Mat &inp, vector<vector<float>> &potential, MTL::Device *device, MetalAdder* adder);
*/

#endif 

#endif