#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "neuron.hpp"

using namespace std;


class Network{
public:
    vector<Neuron> layer2;
    float** synapse;

    int inp_dim;
    int out_dim;

    int inp_x, inp_y, scale, max_time, time_back, time_forwards;
    float p_min, p_rest, w_max, w_min;
    float a_plus, a_minus, tau_plus, tau_minus;

    std::vector<std::vector<float>> kernel;

    Network(int inp_x, int inp_y, 
            int out_dim, 
            int scale = 1, 
            int max_time = 200, 
            int time_back = -20, 
            int time_forwards = 20,
            float p_min = -500,
            float p_rest = 0,
            float w_max = 1.5,
            float w_min = -1.2,
            float a_plus = 0.8,
            float a_minus = 0.3,
            float tau_plus = 8,
            float tau_minus = 5);

    void save_weights(const string filename);
    
    void load_weights(const string filename);

    int lateral_inhibition(vector<int> &active_potential, float thresh);

    void update_weights(vector<vector<float>> &spike_train, int t);

    vector<int> train_on_potential(vector<vector<float>> &potential);

    void reconstruct_weights(float* weights, int num);

    vector<string> get_training_data(const string traindir, const int max_per_category, const bool shuffle=true);

    void reconstruct_all_weights();

    void train(const vector<string> &data_paths, 
                    const int epochs, 
                    const bool verbose = true, 
                    const bool viz_synapse = true);

    int predict(const string filename);


    ~Network();
};

void perform_learning();

#endif