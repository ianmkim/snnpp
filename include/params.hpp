#ifndef PARAMS_H
#define PARAMS_H

namespace Params{
    const int scale = 1;
    const int time = 200;
    const int time_back = -20;
    const int time_forwards = 20;

    // the number of pixels in the X axis
    const int pixel_x = 28;

    // number of neurons in the first layer
    const int num_neurons_1 = pixel_x * pixel_x;
    // number of neurons in the second layer
    const int num_neurons_2 = 3;

    // minimum potential
    const float p_min = -500 * scale;
    // resting potential
    const float p_rest = 0;

    const float w_max = 1.5 * scale;
    const float w_min = -1.2 * scale;
    const float sigma = 0.1;

    const float A_plus = 0.8;
    const float A_minus = 0.3;
    const float tau_plus = 8;
    const float tau_minus = 5;

    const int epoch = 12;

    const int fr_bits = 12;
    const int int_bits = 12;
};

#endif
