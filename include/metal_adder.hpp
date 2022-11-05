#ifndef METAL_ADDER_H
#define METAL_ADDER_H

#ifdef __APPLE__
#include <vector>
#include <opencv2/opencv.hpp>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

class MetalAdder{
public:
    MTL::Device *_mDevice;

    MTL::ComputePipelineState *_mAddFunctionPSO;
    MTL::CommandQueue *_mCommandQueue;

    int arrayLength = 0;
    
    MTL::Buffer *img_buf;
    MTL::Buffer *kernel_buf;
    MTL::Buffer *_mBufferResult;

    MTL::Buffer *img_dim;
    MTL::Buffer *kernel_dim;

    MTL::Buffer *range;

    int img_width;
    int img_height;

    MetalAdder(MTL::Device *device);

    void prepareData(cv::Mat &img, std::vector<std::vector<float>> &kernel);
    void computeReceptiveField(std::vector<std::vector<float>> &potential);
    void verifyResults();

private:
    void encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder);
};

#endif
#endif