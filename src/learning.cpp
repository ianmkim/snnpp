#include <iostream>
#include "math.h"

#include "params.hpp"

/*
 * reinforcement_learning
 * returns the amount a weight should be adjusted by given the time difference
 * between the pre synaptic spike and the post synaptic spike.
 * 
 * @param int time unit of difference between presynaptic and post synaptic spike
 * 
 * @return the amount the weight should change by
 */
float reinforcement_learning(int time){
    if(time > 0){
        return -Params::A_plus * exp(-(float)(time) / Params::tau_plus);
    }
    
    return Params::A_minus * exp((float)(time) / Params::tau_minus);
}

/*
 * stdp_update
 * given a weight and the amount the weight should change by,  stdp_update
 * determines the new value of the weight given certain weight bounds 
 * and the scaling factor
 * 
 * @param float weight
 * @param float change in weight
 * 
 * @return the value of adjusted weight
 */
float stdp_update(float w, float delta_w){
    if(delta_w < 0) {
        return (w + Params::sigma * 
                delta_w *
                (w - abs(Params::w_min)) *
                Params::scale);
    } 

    return (w + Params::sigma *
            delta_w *
            (Params::w_max - w) *
            Params::scale);
}