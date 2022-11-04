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
#include <format>
#include <opencv2/opencv.hpp>

#include "progressbar.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

void reconstruct_weights(vector<float> weights, int num){
    Mat image(Params::pixel_x, Params::pixel_x, CV_8UC1);
    vector<float> r1 = {Params::w_min, Params::w_max};
    vector<float> r2 = {0.0, 255.0};
    for(int i = 0; i < Params::pixel_x; i++){
        for(int j = 0; j < Params::pixel_x; j++){
            uchar pixel_val = (uchar)interpolate(r1, r2, weights.at((Params::pixel_x * i) + j));
            image.at<uchar>(i, j) = pixel_val;
        }
    }
    imwrite("neuron_" + to_string(num) + ".png", image);
}

void perform_learning(){
    // holds potentials of output neurons 
    vector<vector<float>> pot_arrays;
    for(int i = 0; i < Params::num_neurons_2; i++){
        vector<float> pot;
        pot_arrays.push_back(pot);
    }

    // layer2
    vector<Neuron> layer2;

    for(int i = 0; i < Params::num_neurons_2; i++){
        Neuron a;
        layer2.push_back(a);
    }

    vector<vector<float>> synapse(
        Params::num_neurons_2, 
        vector<float>(Params::num_neurons_1, 0));

    for(int i = 0; i < Params::num_neurons_2; i++){
        for(int j = 0; j < Params::num_neurons_1; j++){
            synapse.at(i).at(j) = genRandom(0, 0.4 * Params::scale);
        }
    }

    for(int k = 0; k < Params::epoch; k++){
        cout << "\nProcessing epoch " << k << endl;
        progressbar bar(10 * 10);
        for(int category = 0; category <= 9; category++){
        for(int num = 1; num <= 10; num++){
            bar.update(); 

            //cout << "Processing: " << category << ", " << num << endl;
            string filename = "mnist_set/" + to_string(category) + "/img_" + to_string(num) + ".jpg";

            Mat image = imread(filename, IMREAD_GRAYSCALE);

            vector<vector<float>> potential = produce_receptive_field(image) ;
            vector<vector<float>> spike_train = encode(potential);

            float thresh = threshold(spike_train);
            float var_D = 0.15 * Params::scale;

            for(int i = 0; i < layer2.size(); i++)
                layer2.at(i).initial(thresh);

            bool lateral_inhibition_finished = false;
            int img_win = 100;

            vector<int> active_potential(Params::num_neurons_2, 0);

            // leaky integrate and fire dynamics
            for(int t = 1; t <=Params::time; t++){
                for(int j = 0; j < layer2.size(); j++){
                    Neuron* neuron = &layer2.at(j);
                    if(neuron->t_reset < t){
                        vector<float> sliced = slice_col(t, spike_train);
                        neuron->p += dot<float>(synapse[j], sliced);
                        if(neuron->p > Params::p_rest){
                            neuron->p -= var_D;
                        } active_potential[j] = neuron->p;
                    }

                    pot_arrays[j].push_back(neuron->p);
                }


                if(!lateral_inhibition_finished){
                    float highest_pot = *max_element(active_potential.begin(), active_potential.end());
                    if(highest_pot > thresh){
                        lateral_inhibition_finished = true;
                        int winner = argmax(active_potential);
                        img_win = winner;
                        for(int s = 0; s < Params::num_neurons_2; s++){
                            if(s != winner) layer2[s].p = Params::p_min;
                        }
                    }
                }

                // check for spikes and update weights accordingly
                for(int j = 0; j < layer2.size(); j++){
                    Neuron* neuron = &layer2.at(j);
                    if(neuron->check()){
                        neuron->t_reset = t + neuron->t_refractory;
                        neuron->p = Params::p_rest;
                        for(int h = 0; h < Params::num_neurons_1; h++){
                            for(int t1 = -2; t1 < Params::time_back; t--){
                                if(t+t1 <= Params::time && t+t1 >= 0){
                                    if(spike_train.at(h).at(t+t1) == 1)
                                        synapse.at(j).at(h) = stdp_update(synapse.at(j).at(h), reinforcement_learning(t1));
                                }
                            }


                            for(int t1 = 2; t1 < Params::time_forwards; t1++){
                                if(t+t1 >= 0 && t+t1 <= Params::time){
                                    if(spike_train.at(h).at(t+t1) == 1)
                                        synapse.at(j).at(h) = stdp_update(synapse.at(j).at(h), reinforcement_learning(t1));
                                }
                            }
                        }
                    }
                }
            }
            
            if(img_win != 100){
                for(int p = 0; p < Params::num_neurons_1; p++){
                    if(reduce(spike_train.at(p).begin(), spike_train.at(p).end()) ==0){
                        synapse[img_win][p] -= 0.06 * Params::scale;
                        if(synapse[img_win][p] < Params::w_min){
                            synapse[img_win][p] = Params::w_min;
                        }
                    }
                }
            }
        }
        }
    }


    for(int i = 0; i < Params::num_neurons_2; i++){
        reconstruct_weights(synapse[i], i+1);
    }
}

