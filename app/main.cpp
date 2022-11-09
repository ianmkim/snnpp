#ifdef ENABLE_DOCTEST_IN_LIBRARY
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <stdlib.h>

#include "network.hpp"

using namespace std;

int main(){
    Network net(28, 28, 30);

    vector<string> data_paths = net.get_training_data("mnist_set", 200, true);
    net.train(data_paths, 5, true, true);
    net.save_weights("weights.dat");
    /*
    Network loaded_net(28, 28, 30);
    loaded_net.load_weights("weights.dat");
    loaded_net.reconstruct_all_weights();
    int res = loaded_net.predict("mnist_set/9/img_1.jpg");
    cout << res << endl;
    */
}



