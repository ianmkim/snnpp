#include "neuron.hpp"
#include "params.hpp"

/*
 * Neuron
 * constructor for the neuron class
 */
Neuron::Neuron(){
    this->t_refractory = 30;
    this->t_reset = -1;
    this->p = 0;
    this->p_rest = 0;
    this->p_min = -500;
}

Neuron::Neuron(int t_refractory, int t_reset, float p_rest, float p_min){
    this->t_refractory = t_refractory;
    this->t_reset = t_reset;
    this->p_rest = p_rest;
    this->p_min = p_min;
}

/*
* check
* checks whether the neuron's potential has hit the
* threshold or it hit the minimum potential
*/
bool Neuron::check(){
    if(this->p >= this->p_threshold){
        this->p = this->p_rest;
        return true;
    } else if (this->p < this->p_min){
        this->p = this->p_rest;
        return false;
    }
    return false;
}

/*
* inhibit
* Inhibits the neuron by setting its potential to the
* minimum value it can hold
*/
void Neuron::inhibit(){
    this->p = this->p_min;
}

/*
* initial
* Sets the potential threshold, sets the current potential
* to the resting potential, and time to reset is set so
* that the neuron can reset anytime
*/
void Neuron::initial(const float threshold){
    this->p_threshold = threshold;
    this->t_reset = -1;
    this->p = this->p_rest;
}
