#include "neuron.hpp"
#include "params.hpp"

Neuron::Neuron(){
    this->t_refractory = 30;
    this->t_reset = -1;
    this->p = Params::p_rest;
}


bool Neuron::check(){
    if(this->p >= this->p_threshold){
        this->p = Params::p_rest;
        return true;
    } else if (this->p < Params::p_min){
        this->p = Params::p_rest;
        return false;
    }
    return false;
}


void Neuron::inhibit(){
    this->p = Params::p_min;
}


void Neuron::initial(const float threshold){
    this->p_threshold = threshold;
    this->t_reset = -1;
    this->p = Params::p_rest;
}
