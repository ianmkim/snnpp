#include <iostream>

#include <vector>
#include <opencv2/opencv.hpp>

#include "params.hpp"
#include "receptive_field.hpp"

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "metal_adder.hpp"
#endif

using namespace std;

#ifdef __APPLE__
void produce_receptive_field_metal(cv::Mat &inp, vector<vector<float>> &potential){
    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3}, 
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };

    MetalAdder *adder = new MetalAdder(device);

    adder->prepareData(inp, kernel);
    adder->computeReceptiveField(potential);
}
#endif

vector<vector<float>> produce_receptive_field(cv::Mat &inp){
    vector<vector<float>> potential(
        inp.rows, 
        vector<float>(inp.cols, 0.0));

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3}, 
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };

    vector<int> ran {-2, -1, 0, 1, 2};
    int ox = 2;
    int oy = 2;

    for(int i = 0; i < potential.size(); i++){
        for(int j = 0; j < potential[0].size(); j++){
            float sum = 0;

            for(int m : ran){
                for(int n : ran){
                    if((i+m) >= 0 && 
                        (i+m) <= Params::pixel_x-1 && 
                        (j+n) >= 0 &&
                        (j+n) <= Params::pixel_x-1){
                        uchar pixel_uc = inp.at<uchar>(i+m, j+n);
                        float pixel = (float)pixel_uc  ;
                        pixel /= 255.0;
                        sum += kernel.at(ox+m).at(oy+n) * 
                                pixel;
                    }
                }
            }

            potential[i][j] = sum;
        }
    }
    
    /*
    cv::Mat image(potential.size(), potential.at(0).size(), CV_64FC1);
    for(int i = 0; i<image.rows; ++i){
        for(int j = 0; j <image.cols;j++){
            image.at<double>(i, j) = potential.at(i).at(j) * 255.0;
        }
    }
    imwrite("something.png", image);
    */
    

    return potential;
}