#include "doctest/doctest.h"

#include <iostream>

#include "neuron.hpp"


TEST_CASE("Neuron class initializing"){
    Neuron neuron = Neuron();
    neuron.initial(0.1);
    CHECK(neuron.check() == false);
}


TEST_CASE("Dynamically allocating Neuron class"){
    Neuron *neuron_ptr = new Neuron();
    neuron_ptr->initial(0.1);
    CHECK(neuron_ptr->check() == false);
    delete neuron_ptr;
}
