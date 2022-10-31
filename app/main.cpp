#ifdef ENABLE_DOCTEST_IN_LIBRARY
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <stdlib.h>

#include "neuron.hpp"


int main(){
    std::cout << "Spiking Neural Network" << std::endl;

    Neuron neuron = Neuron();
    std::cout << "Neuron check res: " << neuron.check() << std::endl;
}



