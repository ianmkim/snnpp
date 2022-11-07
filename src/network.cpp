#include "neuron.hpp"
#include "receptive_field.hpp"
#include "spike_train.hpp"
#include "learning.hpp"
#include "utils.hpp"
#include "threshold.hpp"

#include <stdio.h>

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "progressbar.hpp"
#include "utils.hpp"
#include "network.hpp"

using namespace cv;
using namespace std;

Network::Network(int inp_x, int inp_y, 
            int out_dim, 
            int scale, 
            int max_time, 
            int time_back, 
            int time_forwards,
            float p_min,
            float p_rest,
            float w_max,
            float w_min,
            float a_plus,
            float a_minus,
            float tau_plus,
            float tau_minus){
    this->inp_dim = inp_x * inp_y;
    this->out_dim = out_dim;

    this->inp_x = inp_x;
    this->inp_y = inp_y;
    
    this->scale = scale;
    this->max_time = max_time;
    this->time_back = time_back;
    this->time_forwards = time_forwards;
    
    this->p_min = p_min;
    this->p_rest = p_rest;
    
    this->w_max = w_max;
    this->w_min = w_min;
    
    this->a_plus = a_plus;
    this->a_minus = a_minus;

    float field1 = 0.625;
    float field2 = 0.125;
    float field3 = -0.125;
    float field4 = -0.5;

    vector<vector<float>> kernel {
        {field4, field3, field2, field3, field4},
        {field3, field2, field1, field2, field3},
        {field2, field1,    1.0, field1, field2},
        {field3, field2, field1, field2, field3},
        {field4, field3, field2, field3, field4},
    };

    this->kernel = kernel;

    for(int i = 0; i < out_dim; i++){
        Neuron a;
        this->layer2.push_back(a);
    }

    this->synapse = (float**)malloc(out_dim * sizeof(float*));
    for(int i = 0; i < out_dim; i++){
        this->synapse[i] = (float*)malloc(inp_dim * sizeof(float));
        memset(this->synapse[i], (float)0.0, inp_dim * sizeof(float));
    }

    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim ; j++){
            synapse[i][j] = genRandom(0, 0.4 * this->scale);
        }
    }
}

void Network::save_weights(const string filename){
    FILE *dat = fopen(filename.c_str(), "w"); // opens new file for writing
    if(dat == NULL) return;
    
    fprintf(dat, "%d %d %d\n", this->inp_x, this->inp_y, this->out_dim);
    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim; j++){
            fprintf(dat, "%f ", this->synapse[i][j]);
        }
        fprintf(dat, "\n");
    }
}

void Network::load_weights(const string filename){
    FILE *dat = fopen(filename.c_str(), "r");
    if(dat == NULL) return;

    int r_inp_x, r_inp_y, r_out_dim;
    fscanf(dat, "%d %d %d\n", &r_inp_x, &r_inp_y, &r_out_dim);
    if(r_inp_x != this->inp_x || r_inp_y != this->inp_x || r_out_dim != out_dim){
        cerr << "Loading weights failed because the input and output dimensions don't match" << endl;
        return;
    }

    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim; j++){
            fscanf(dat, "%f", &this->synapse[i][j]);
        }
    }
}

int Network::lateral_inhibition(vector<int> &active_potential, float thresh) {
    float highest_pot = *max_element(active_potential.begin(), active_potential.end());
    int img_win = -1;
    if(highest_pot > thresh){
        int winner = argmax(active_potential);
        img_win = winner;
        for(int s = 0; s < this->out_dim; s++){
            if(s != winner) this->layer2[s].p = this->p_min;
        }
    }
    return img_win;
}

void Network::update_weights(vector<vector<float>> &spike_train, int t){
    for(int j = 0; j < this->layer2.size(); j++){
        Neuron* neuron = &this->layer2.at(j);
        if(neuron->check()){
            neuron->t_reset = t + neuron->t_refractory;
            neuron->p = this->p_rest;
            for(int h = 0; h < this->inp_dim; h++){
                for(int t1 = -2; t1 < this->time_back; t--){
                    if(t+t1 <= this->max_time && t+t1 >= 0){
                        if(spike_train.at(h).at(t+t1) == 1)
                            synapse[j][h] = stdp_update(synapse[j][h], reinforcement_learning(t1));
                    }
                }


                for(int t1 = 2; t1 < this->time_forwards; t1++){
                    if(t+t1 >= 0 && t+t1 <= this->max_time){
                        if(spike_train.at(h).at(t+t1) == 1)
                            synapse[j][h] = stdp_update(synapse[j][h], reinforcement_learning(t1));
                    }
                }
            }
        }
    }
}

vector<int> Network::train_on_potential(vector<vector<float>> &potential){
    vector<int> num_spikes(this->out_dim);

    vector<vector<float>> spike_train = encode(potential);

    float thresh = threshold(spike_train);
    float var_D = 0.15 * this->scale;

    for(int i = 0; i < this->layer2.size(); i++)
        this->layer2.at(i).initial(thresh);

    bool lateral_inhibition_finished = false;
    int img_win = -1;

    vector<int> active_potential(this->out_dim, 0);

    for(int t = 1; t <= this->max_time; t++){
        for(int j = 0; j < this->layer2.size(); j++){
            Neuron* neuron = &this->layer2.at(j);
            if(neuron->t_reset < t){
                vector<float> sliced = slice_col(t, spike_train);
                neuron->p += dot_sse(synapse[j], sliced.data(), sliced.size());
                if(neuron->p > this->p_rest){
                    neuron->p -= var_D;
                } active_potential[j] = neuron->p;
            }
        }


        if(!lateral_inhibition_finished){
            int winner = this->lateral_inhibition(active_potential, thresh);
            if(winner != -1){
                img_win = winner;
                lateral_inhibition_finished = true;
            }
        }

        for(int n_indx = 0; n_indx < this->out_dim; n_indx++)
            if(this->layer2.at(n_indx).check())
                num_spikes[n_indx]++;

        // check for spikes and update weights accordingly
        this->update_weights(spike_train, t);
        
    }
    
    if(img_win != -1){
        for(int p = 0; p < Params::num_neurons_1; p++){
            if(reduce(spike_train.at(p).begin(), spike_train.at(p).end()) == 0){
                synapse[img_win][p] -= 0.06 * Params::scale;
                if(synapse[img_win][p] < Params::w_min){
                    synapse[img_win][p] = Params::w_min;
                }
            }
        }
    }

    return num_spikes;
}

void Network::reconstruct_weights(float* weights, int num){
    Mat image(this->inp_x, this->inp_y, CV_8UC1);
    vector<float> r1 = {this->w_min, this->w_max};
    vector<float> r2 = {0.0, 255.0};
    for(int i = 0; i < this->inp_x; i++){
        for(int j = 0; j < this->inp_y; j++){
            uchar pixel_val = (uchar)interpolate(r1, r2, weights[(Params::pixel_x * i) + j]);
            image.at<uchar>(i, j) = pixel_val;
        }
    }
    imwrite("neuron_" + to_string(num) + ".png", image);
}

vector<string> Network::get_training_data(const string traindir, 
                                    const int max_per_category, 
                                    const bool shuffle){
    vector<string> dirs;
    for(auto& p : std::filesystem::directory_iterator(traindir))
        if (p.is_directory())
            dirs.push_back(p.path().string());

    vector<string> paths_to_data;
    if(dirs.size() != 0){
        for(auto& dir : dirs){
            int cat = 0;
            for(auto& p : std::filesystem::directory_iterator(dir)){
                if(p.is_regular_file() && cat < max_per_category){
                    paths_to_data.push_back(dir + "/" + p.path().filename().string());
                    cat++;
                }
            }
        }

    } else{
        for(auto& p: std::filesystem::directory_iterator(traindir))
            if(p.is_regular_file())
                paths_to_data.push_back(traindir + "/" + p.path().filename().string());
    }

    if(shuffle){
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(paths_to_data), std::end(paths_to_data), rng);
    }
    return paths_to_data;
}

void Network::reconstruct_all_weights(){
    for(int i = 0; i < this->out_dim; i++){
        this->reconstruct_weights(this->synapse[i], i+1);
    }
}

void Network::train(const vector<string> &data_paths, 
                    const int epochs, 
                    const bool verbose, 
                    const bool viz_synapse){

    for(int k = 0; k < epochs; k++){
        cout << "\nProcessing epoch " << k << endl;
        progressbar bar(data_paths.size());
        
        for(string filename : data_paths){
            Mat image = imread(filename, IMREAD_GRAYSCALE);
            if(!image.empty()){
                if(verbose) bar.update(); 
                vector<vector<float>> potential = produce_receptive_field(image, this->kernel);
                this->train_on_potential(potential);
            }
        }
    }

    if(viz_synapse) this->reconstruct_all_weights();
}

int Network::predict(const string filename){
    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if(!image.empty()){
        vector<vector<float>> potential = produce_receptive_field(image, this->kernel);
        vector<int> spikes_per_neuron = this->train_on_potential(potential);
        std::vector<int>::iterator max = max_element(spikes_per_neuron.begin(), spikes_per_neuron.end()); // [2, 4)
        return distance(spikes_per_neuron.begin(), max);
    }
    return -1;
}


Network::~Network(){
    for(int i = 0; i < this->out_dim; i++){
        free(this->synapse[i]);
    } free(this->synapse);
}

void perform_learning(){
    Network net(28, 28, 30);
    /*
    vector<string> data_paths = net.get_training_data("mnist_set", 500, true);
    net.train(data_paths, 12, true, true);
    net.save_weights("weights.dat");
    */

    net.load_weights("weights.dat");
    net.reconstruct_all_weights();
    int res = net.predict("mnist_set/9/img_1.jpg");
    cout << res << endl;
}

