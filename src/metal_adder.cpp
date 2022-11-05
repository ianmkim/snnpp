#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "metal_adder.hpp"
#include "receptive_field.hpp"

#ifdef __APPLE__
MetalAdder::MetalAdder(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr)
    {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
    MTL::Function *addFunction = defaultLibrary->newFunction(str);

    if (addFunction == nullptr)
    {
        std::cout << "Failed to find the adder function." << std::endl;
        return;
    }

    // Create a compute pipeline state object.
    _mAddFunctionPSO = _mDevice->newComputePipelineState(addFunction, &error);

    if (_mAddFunctionPSO == nullptr)
    {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
}

void MetalAdder::prepareData(cv::Mat &img, std::vector<std::vector<float>> &kernel)
{
    assert(img.channels() == 1);
    assert(img.isContinuous());

    range = _mDevice->newBuffer(5, MTL::ResourceStorageModeShared);
    int* range_ptr = (int*)range->contents();
    range_ptr[0] = -2;
    range_ptr[1] = -1;
    range_ptr[2] =  0;
    range_ptr[3] =  1;
    range_ptr[4] =  2;

    // Allocate three buffers to hold our initial data and the result.
    img_buf = _mDevice->newBuffer(img.cols * img.rows, MTL::ResourceStorageModeShared);
    img_width = img.cols;
    img_height = img.rows;
    kernel_buf = _mDevice->newBuffer(kernel.size() * kernel.at(0).size(), MTL::ResourceStorageModeShared);

    img_dim = _mDevice->newBuffer(2, MTL::ResourceStorageModeShared);
    kernel_dim = _mDevice->newBuffer(2, MTL::ResourceStorageModeShared);

    _mBufferResult = _mDevice->newBuffer(img.cols * img.rows, MTL::ResourceStorageModeShared);

    int* img_dim_ptr = (int*)img_dim->contents();
    img_dim_ptr[0] = img.cols;
    img_dim_ptr[1] = img.rows;

    int* kernel_dim_ptr = (int*)kernel_dim->contents();
    kernel_dim_ptr[0] = kernel.size();
    kernel_dim_ptr[1] = kernel.at(0).size();

    unsigned char *data_ptr = (unsigned char*)img_buf->contents();
    std::memcpy(data_ptr, img.data, img.cols * img.rows * sizeof(unsigned char));

    float* kernel_ptr = (float*)kernel_buf->contents();
    int idx = 0;
    for(int i = 0; i < kernel.size(); i++){
        for(int j = 0; j < kernel.size(); j++){
            kernel_ptr[idx++] = kernel.at(i).at(j);
        }
    }

    this->arrayLength = img.cols * img.rows;
}

void MetalAdder::computeReceptiveField(std::vector<std::vector<float>> &potential)
{
    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    encodeAddCommand(computeEncoder);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
    float *result = (float*)_mBufferResult->contents();
    
    for(int i = 0; i < potential.size(); i++)
        memcpy(potential.at(i).data(), result + (i * potential.size()), sizeof(float)*potential.at(0).size());
}

void MetalAdder::encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder)
{
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mAddFunctionPSO);
    computeEncoder->setBuffer(img_buf,        0, 0);
    computeEncoder->setBuffer(kernel_buf,     0, 1);
    computeEncoder->setBuffer(img_dim,        0, 2);
    computeEncoder->setBuffer(kernel_dim,     0, 3);
    computeEncoder->setBuffer(range,          0, 4);
    computeEncoder->setBuffer(_mBufferResult, 0, 5);

    //MTL::Size gridSize = MTL::Size::Make(img_width, img_height, 1);

    // Calculate a threadgroup size.
    NS::UInteger w = _mAddFunctionPSO->threadExecutionWidth();
    NS::UInteger h = _mAddFunctionPSO->maxTotalThreadsPerThreadgroup()/w;
    MTL::Size threadsPerGroup = MTL::Size::Make(w, h, 1);
    MTL::Size threadsPerGrid = MTL::Size::Make(img_width, img_height, 1);
    MTL::Size threadgroupsPerGrid = MTL::Size::Make((img_width + w - 1) / w,
                                                    (img_height + h - 1) / h,
                                                    1);


    /*
    int group_width = img_width;
    int group_height = img_height;
    if(img_width * img_height > threadGroupSize){
        group_width = img_height/32; 
        group_height = img_height/32;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(group_width, group_height, 1);

    std::cout << "Grid size: " << img_width << " " << img_height << std::endl;
    */

    // Encode the compute command.
    computeEncoder->dispatchThreads(threadsPerGrid, threadsPerGroup);

    //computeEncoder->dispatchThreadgroups(threadgroupsPerGrid, threadsPerGroup);
}


void MetalAdder::verifyResults()
{
    float *result = (float *)_mBufferResult->contents();
    cv::Mat image(img_width, img_height, CV_64FC1);
    int idx = 0;
    for(int i = 0; i< img_height; ++i){
        for(int j = 0; j < img_width;j++){
            float res = result[idx++];
            image.at<double>(i, j) = res * 255.0;
        }
    }
    imwrite("GPU_SHADER_OUT.png", image);
}
#endif