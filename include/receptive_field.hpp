#ifndef RECEPTIVE_FIELD_H
#define RECEPTIVE_FIELD_H
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

/*
 * produce_receptive_field_sse
 * produces a receptive field given a reference to the input image
 * by first padding the image, then using an on-center 5x5 kernel
 * we convolve the image to produce the membrane potential which 
 * we write to an already allocated potential array. 
 * 
 * This function is the SSE optimized version of the other
 * produce_receptive_field_sse functions.
 * 
 * THE CALLER IS RESPONSIBLE FOR ALLOCATING ENOUGH SPACE TO HOLD
 * THE POTENTIAL WHICH SHOULD BE THE SAME DIMENSIONS AS THE INPUT IMAGE
 * 
 * @param cv mat image
 * @param 2d kernel to convolve the image with
 * @param pre allocated potential 2d array that has same dims as the image
 */
void produce_receptive_field_sse(cv::Mat &inp, 
                                const vector<vector<float>> &kernel, 
                                vector<vector<float>> &potential);

/*
 * produce_receptive_field
 * produces a receptive field given a reference to the input image
 * by first padding the image, then using an on-center 5x5 kernel
 * we convolve the image to produce the membrane potential which 
 * we write to an already allocated potential array. 
 * 
 * @param cv mat image
 * @param 2d kernel to convolve the image with
 */
vector<vector<float>> produce_receptive_field(cv::Mat &inp, vector<vector<float>> &kernel);


#ifdef __APPLE__
//vector<vector<float>> produce_receptive_field_metal(cv::Mat &inp);
/*
void prepare_metal_device(MTL::Device *device, MetalAdder* adder);
void produce_receptive_field_metal(cv::Mat &inp, vector<vector<float>> &potential, MTL::Device *device, MetalAdder* adder);
*/
#endif 

#endif