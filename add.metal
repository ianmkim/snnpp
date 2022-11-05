#include <metal_stdlib>
using namespace metal;

/*
kernel int index_to_x(device int idx, device int width){
    return (int)(idx%width);
}

kernel int index_to_y(device int idx, device int height){
    return (int)(idx/height);
}

kernel int xy_to_index(device int x, device int y, device int width){
    return (y * width) + x;
}
*/

/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays()
/// C function, used to perform the calculation on a GPU.
kernel void add_arrays(device const uchar* img,
                       device const float* kernel_buf,
                       device const int* imgDim,
                       device const int* kernelDim,
                       device const int* range,
                       device float* result,
                       uint2 tid [[thread_position_in_grid]] )
{
    int width = imgDim[0];
    int height = imgDim[1];

    int kernel_width = kernelDim[0];

    int x = tid.x;
    int y = tid.y;

    if(x >= width || y >= height){
        return;
    }

    int ox = 2;
    int oy = 2;

    float sum = 0.0;

    for(int i = 0; i < 5; i++){
        int m = range[i];
        for(int j = 0; j < 5; j++){
            int n = range[j];
            
            int kernel_x = ox + m;
            int kernel_y = oy + n;
            
            if((x+m)  >= 0    &&
                (x+m) < width &&
                (y+n) >= 0    &&
                (y+n) < height){
                sum += kernel_buf[(kernel_y * kernel_width) + kernel_x] * 
                        (((float)img[((y+n)*width)+(x+m)]) / 255.0);
            }
        }
    }

    result[(y*width) + x] = sum;
}