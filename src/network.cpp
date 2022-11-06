#include "neuron.hpp"
#include "receptive_field.hpp"
#include "params.hpp"
#include "spike_train.hpp"
#include "learning.hpp"
#include "utils.hpp"
#include "threshold.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "progressbar.hpp"
#include "utils.hpp"

using namespace cv;
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
            float tau_minus = 5){
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

    int lateral_inhibition(vector<int> &active_potential, float thresh) {
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

    void update_weights(vector<vector<float>> &spike_train, int t){
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

    void train_on_potential(vector<vector<float>> &potential){
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
    }

    void reconstruct_weights(float* weights, int num){
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

    vector<string> get_training_data_paths(string traindir){
        vector<string> dirs;
        for(auto& p : std::filesystem::directory_iterator(traindir))
            if (p.is_directory())
                dirs.push_back(p.path().string());
        return dirs;
    }

    void train(int epochs, bool verbose=false, bool viz_synapse=false){
        for(int k = 0; k < epochs; k++){
            cout << "\nProcessing epoch " << k << endl;
            progressbar bar(10 * 10);

            for(int category = 0; category <= 9; category++){
                for(int num = 1; num <= 10; num++){
                    if(verbose) bar.update(); 

                    string filename = "mnist_set/" + to_string(category) + "/img_" + to_string(num) + ".jpg";

                    Mat image = imread(filename, IMREAD_GRAYSCALE);

                    vector<vector<float>> potential = produce_receptive_field(image);
                    this->train_on_potential(potential);
                }
            }
        }

        for(int i = 0; i < this->out_dim; i++){
            reconstruct_weights(this->synapse[i], i+1);
        } 
    }

    ~Network(){
        for(int i = 0; i < this->out_dim; i++){
            free(this->synapse[i]);
        } free(this->synapse);
    }

};


void perform_learning(){
    Network net(28, 28, 30);
    net.train(5, true, true);
}

