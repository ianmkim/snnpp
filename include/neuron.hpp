#ifndef NEURON_H
#define NEURON_H

class Neuron{
public:
    // refractory period
    int t_refractory;
    // time to reset neuron
    int t_reset;
    // current potential
    float p;
    // potential threshold
    float p_threshold;

    // resting potential and min potential
    float p_rest;
    float p_min;

    /*
     * Neuron
     * constructor for the neuron class,
     * sets the refractory time, time to reset and
     * the current potential as the resting potential
     */
    Neuron();

    /*
     * Neuron
     * constructor for the neuron class,
     * sets the refractory time, time to reset and
     * the current potential as the resting potential
     */
    Neuron(int t_refractory, int t_reset, float p_rest, float p_min);

    /*
     * check
     * checks whether the neuron's potential has hit the
     * threshold or it hit the minimum potential
     */
    bool check();

    /*
     * inhibit
     * Inhibits the neuron by setting its potential to the
     * minimum value it can hold
     */
    void inhibit();

    /*
     * initial
     * Sets the potential threshold, sets the current potential
     * to the resting potential, and time to reset is set so
     * that the neuron can reset anytime
     */
    void initial(const float threshold);
};


#endif
