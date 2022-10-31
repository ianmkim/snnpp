#include <iostream>
#include "math.h"

#include "params.hpp"

float reinforcement_learning(int time){
    if(time > 0){
        return -Params::A_plus * exp(-(float)(time) / Params::tau_plus);
    }
    
    return Params::A_minus * exp((float)(time) / Params::tau_minus);
}

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