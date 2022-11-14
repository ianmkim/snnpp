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
            float tau_minus,
            bool simd_optimized){
    // input and output dimensions of the network
    // input dimensions are length of flattened image
    this->inp_dim = inp_x * inp_y;
    // output dimensions are number of output neurons (must be at least
    // as many as the desired number of classes)
    this->out_dim = out_dim;

    // in case the x and y dimensions of the image are different from eachother
    this->inp_x = inp_x;
    this->inp_y = inp_y;

    this->scale = scale;
    // max time to simulate the electric flow through the network
    this->max_time = max_time;
    // amount of time units to look back and forward while training
    this->time_back = time_back;
    this->time_forwards = time_forwards;

    // absolute minimum and resting potentials
    this->p_min = p_min;
    this->p_rest = p_rest;

    // absolute max and min values of synaptic weights
    this->w_max = w_max;
    this->w_min = w_min;

    // constants for negative and positive values of time difference
    // between post synaptic and pre synaptic spikes, determining the max
    // excitation and inhibition values
    // pp.6 Iakymchuk et al.
    this->a_plus = a_plus;
    this->a_minus = a_minus;

    // kernel for off centered and on centered receptive field
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

    // initialize the output neurons
    for(int i = 0; i < out_dim; i++){
        Neuron a;
        this->layer2.push_back(a);
    }

    // allocate memory on the heap for the synapse.
    // using a 2d array is leagues faster than using a 2d vector
    this->synapse = (float**)malloc(out_dim * sizeof(float*));
    for(int i = 0; i < out_dim; i++){
        // allocate the inner layers
        this->synapse[i] = (float*)malloc(inp_dim * sizeof(float));
        memset(this->synapse[i], (float)0.0, inp_dim * sizeof(float));
    }

    // assign random starting weights
    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim ; j++){
            synapse[i][j] = genRandom(0, 0.4 * this->scale);
        }
    }
}

/*
 * save_weights
 * utility function to save the weights to disk
 *
 * @param string filename to save the weights to
 */
void Network::save_weights(const string filename){
    FILE *dat = fopen(filename.c_str(), "w"); // opens new file for writing
    if(dat == NULL) return;

    // write dimensions to the file first so that when we open the file
    // we can check whether the dimensions are the same or not
    fprintf(dat, "%d %d %d\n", this->inp_x, this->inp_y, this->out_dim);
    // save the weights as a 2d grid so that we can do one to one mappings
    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim; j++){
            fprintf(dat, "%f ", this->synapse[i][j]);
        }
        fprintf(dat, "\n");
    }
}

/*
 * load_weights
 * utility function to load weights from disk
 *
 * @param string filepath to load the weights from
 */
void Network::load_weights(const string filename){
    FILE *dat = fopen(filename.c_str(), "r");
    if(dat == NULL) return;

    // read the input data
    int r_inp_x, r_inp_y, r_out_dim;
    fscanf(dat, "%d %d %d\n", &r_inp_x, &r_inp_y, &r_out_dim);
    // check whether the weights were saved in a campatible shape
    if(r_inp_x != this->inp_x || r_inp_y != this->inp_x || r_out_dim != out_dim){
        cerr << "Loading weights failed because the input and output dimensions don't match" << endl;
        return;
    }

    // set the synaptic weights
    for(int i = 0; i < this->out_dim; i++){
        for(int j = 0; j < this->inp_dim; j++){
            fscanf(dat, "%f", &this->synapse[i][j]);
        }
    }
}

/*
 * lateral inhibition
 * Given a threshold and the active potentials of the final layer, performs
 * lateral inhibition such that inhibitory signals are sent to all other neurons
 * in the final layers and only the most active neuron gets his weights adjusted
 *
 * However, there is a threshold term to eliminate only minutely different
 *
 * @param vector of ints that indicate the active potentials
 * @param float minimum threshold to be considered a winner
 *
 * @return int the index of the neuron in the final layer that won for this image
 */
int Network::lateral_inhibition(vector<int> &active_potential, float thresh) {
    // get the highest potential out of all the active potentials
    float highest_pot = *max_element(active_potential.begin(), active_potential.end());
    int img_win = -1;
    // if the highest activation is at least as large as the threshold
    if(highest_pot > thresh){
        int winner = argmax(active_potential);
        img_win = winner;
        // inhibit all the loser neurons in the layer
        for(int s = 0; s < this->out_dim; s++){
            if(s != winner) this->layer2[s].p = this->p_min;
        }
    }
    return img_win;
}

/*
 * update_weights
 * given a the spike patterns of all neurons in the final layer and the current
 * time unit in the simulation, update the weights from the input to output
 * neurons accordingly
 *
 * @param 2d vector of floats that indicate spike values
 * @param int time
 */
void Network::update_weights(vector<vector<float>> &spike_train, int t){
    // iterate through all the neurons in the output layers
    for(int j = 0; j < this->layer2.size(); j++){
        Neuron* neuron = &this->layer2.at(j);
        // check whether the neuron's potential has hit the threshold
        if(neuron->check()){
            // make sure that the neuron doesn't fire until it has finished waiting
            // until the refractory period
            neuron->t_reset = t + neuron->t_refractory;
            // reset potential to the resting potential
            neuron->p = this->p_rest;

            // loop over all weights
            for(int h = 0; h < this->inp_dim; h++){
                for(int t1 = -2; t1 < this->time_back; t--){
                    // if the look back is within the bounds of the simulation
                    if(t+t1 <= this->max_time && t+t1 >= 0){
                        // if it sppiked within the time bounds, then it means that the spike
                        // from this synapse probably contributed to this neuron, so update the
                        // weights
                        if(spike_train.at(h).at(t+t1) == 1)
                            synapse[j][h] = stdp_update(synapse[j][h], reinforcement_learning(t1));
                    }
                }


                // do the same thing, except for the forward times
                for(int t1 = 2; t1 < this->time_forwards; t1++){
                    if(t+t1 >= 0 && t+t1 <= this->max_time){
                        // we want to decrease influence for these ones now
                        if(spike_train.at(h).at(t+t1) == 1)
                            synapse[j][h] = stdp_update(synapse[j][h], reinforcement_learning(t1));
                    }
                }
            }
        }
    }
}

/*
 * train_on_potential
 * adjust the weights of the network for one simulation run (0-200 time units)
 * based on one single input image (which must be preprocessed into membrane
 * potential)
 *
 * @param potential
 * @return the number of spikes of every neuron
 */
vector<int> Network::train_on_potential(vector<vector<float>> &potential){
    vector<int> num_spikes(this->out_dim);
    // encode the potential produced by running it through the receptive field
    // into a spike train so that it can be given as input to the network
    vector<vector<float>> spike_train = encode(potential);

    // get a custom threshold for the given image
    float thresh = threshold(spike_train);
    float var_D = 0.15 * this->scale;

    for(int i = 0; i < this->layer2.size(); i++)
        this->layer2.at(i).initial(thresh);

    // we only want to do lateral inhibition once per time
    bool lateral_inhibition_finished = false;
    int img_win = -1;

    vector<int> active_potential(this->out_dim, 0);

    // simulate the network through the entire time period
    for(int t = 1; t <= this->max_time; t++){
        // iterate through all the output neurons
        for(int j = 0; j < this->layer2.size(); j++){
            Neuron* neuron = &this->layer2.at(j);

            if(neuron->t_reset < t){
                // simulate the potential accumulation in the neuron
                // by doing a dot product of the input spike train with the
                // synaptic weights for the current neuron
                vector<float> sliced = slice_col(t, spike_train);
                neuron->p += dot_sse(synapse[j], sliced.data(), sliced.size());

                // if the potential is greater than resting potential
                if(neuron->p > this->p_rest){
                    // decrease the potential
                    neuron->p -= var_D;
                } active_potential[j] = neuron->p;
            }
        }

        // perform lateral inhibition if it has not been performed already
        if(!lateral_inhibition_finished){
            int winner = this->lateral_inhibition(active_potential, thresh);
            if(winner != -1){
                img_win = winner;
                lateral_inhibition_finished = true;
            }
        }

        // for every neuron in the output layer, note how many times each neuron
        // spiked up to this point in time
        for(int n_indx = 0; n_indx < this->out_dim; n_indx++)
            if(this->layer2.at(n_indx).check())
                num_spikes[n_indx]++;

        // check for spikes and update weights accordingly
        this->update_weights(spike_train, t);

    }

    // if there was a winner neuron
    if(img_win != -1){
        for(int p = 0; p < this->inp_dim; p++){
            // if there were no spikes at all from this particular input neuron
            if(reduce(spike_train.at(p).begin(), spike_train.at(p).end()) == 0){
                // decrease the synaptic weights
                synapse[img_win][p] -= 0.06 * this->scale;
                // make sure that the weight doesn't go below minimum weight
                if(synapse[img_win][p] < this->w_min){
                    synapse[img_win][p] = this->w_min;
                }
            }
        }
    }

    // return the total number of spikes for each neuron
    return num_spikes;
}

/*
 * reconstruct_weights
 * given all weights of a single neuron, it reshapes it as an image
 * and saves it. Because of the single layer archetecture, we can directly
 * reconstruct weights to use as generated digits.
 *
 * @param float array of weights
 * @param int number of the neuron
 */
void Network::reconstruct_weights(float* weights, int num){
    Mat image(this->inp_x, this->inp_y, CV_8UC1);
    vector<float> r1 = {this->w_min, this->w_max};
    // max value for a pixel in the image is 255 since 8 bits unsigned char
    vector<float> r2 = {0.0, 255.0};
    for(int i = 0; i < this->inp_x; i++){
        for(int j = 0; j < this->inp_y; j++){
            // perform linear interpolation to scale the weights to the pixel values
            uchar pixel_val = (uchar)interpolate(r1, r2, weights[(Params::pixel_x * i) + j]);
            // set the pixel value
            image.at<uchar>(i, j) = pixel_val;
        }
    }

    // write the image to disk
    imwrite("neuron_" + to_string(num) + ".png", image);
}

/*
 * reconstruct_all_weights
 * reconstruct the weights of all neurons in the final layer
 */
void Network::reconstruct_all_weights(){
    for(int i = 0; i < this->out_dim; i++){
        this->reconstruct_weights(this->synapse[i], i+1);
    }
}

void Network::reconstruct_all_weights_for_animation(const string filepath, int num){
    for(int n = 0; n < this->out_dim; n++) {
        auto weights = this->synapse[n];
        Mat image(this->inp_x, this->inp_y, CV_8UC1);
        vector<float> r1 = {this->w_min, this->w_max};
        vector<float> r2 = {0.0, 255.0};
        for(int i = 0; i < this->inp_x; i++){
            for(int j = 0; j < this->inp_y; j++){
                uchar pixel_val = (uchar)interpolate(r1, r2, weights[(Params::pixel_x * i) + j]);
                image.at<uchar>(i, j) = pixel_val;
            }
        }
        string out_path = filepath + "/" + to_string(n) + "/" + to_string(num) + ".png";
        imwrite(out_path, image);
    }
}

/*
 * get_training_data
 * given a path to the training directory, it retrieves and returns
 * a vector of image paths that comprise the training set. The training
 * directory could either be categorized or uncategorized.
 *
 *
 * 1) categorized:
 * minst_set/
 *   category1/
 *       img1.jpg
 *       img2.jpg
 *       ...
 *   category2/
 *       img1.jpg
 *       img2.jpg
 *       ...
 *   ...
 *
 * 2) uncategorized:
 * mnist_set/
 *    img1.jpg
 *    img2.jpg
 *    img3.jpg
 *    img4.jpg
 *    ...
 *
 * Optionally you can also choose how many pieces of data to retrive from each category
 *
 * @param constant string path to the data directory
 * @param constant int for how many pieces of data per category
 * @param constant boolean for whether to shuffle the dataset
 *
 * @return vector of strings for image paths
 */
vector<string> Network::get_training_data(const string traindir,
                                    const int max_per_category,
                                    const bool shuffle){
    // get all directories in the training directories
    vector<string> dirs;
    for(auto& p : std::filesystem::directory_iterator(traindir))
        if (p.is_directory())
            dirs.push_back(p.path().string());

    vector<string> paths_to_data;
    // if in categorized form
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

    }

    // if in uncategorizedf form
    else{
        for(auto& p: std::filesystem::directory_iterator(traindir))
            if(p.is_regular_file())
                paths_to_data.push_back(traindir + "/" + p.path().filename().string());
    }

    // shuffle the dataset if requested
    if(shuffle){
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(paths_to_data), std::end(paths_to_data), rng);
    }
    return paths_to_data;
}



/*
 * train
 * trains the neural network on the given set of images for n
 * epochs. You can pass in additional options for verbosity
 * and whether to vizualize the output synapses
 *
 * @param vector of paths to training images, you can get this from get_training_data()
 * @param int epochs to train the network
 * @param boolean enables and disables the progress bar
 * @param boolean determines whether to reconstruct weights into images
 */
void Network::train(const vector<string> &data_paths,
                    const int epochs,
                    const bool verbose,
                    const bool viz_synapse){
    int indx = 0;
    for(int k = 0; k < epochs; k++){
        cout << "\nProcessing epoch " << k << endl;
        progressbar bar(data_paths.size());
        for(string filename : data_paths){
            Mat image = imread(filename, IMREAD_GRAYSCALE);
            if(!image.empty()){
                if(verbose) bar.update();

                // convert the raw image into potential
                vector<vector<float>> potential(image.rows, vector<float>(image.cols, 0));
                produce_receptive_field_sse(image, this->kernel, potential);

                // train the network on that potential
                this->train_on_potential(potential);
            }

            //if(viz_synapse) this->reconstruct_all_weights_for_animation("animation", indx++);
        }
    }

    if(verbose) cout << endl;
    if(viz_synapse) this->reconstruct_all_weights();
}

/*
 * predict
 * predicts what category a single given image will be in by doing a forward pass
 * simulation through the network.
 *
 * @param constant string filepath to the input image
 *
 * @return int index of the output neuron with most activation. -1 if no image given
 */
int Network::predict(const string filename){
    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if(!image.empty()){
        // convert image into membrane potential
        vector<vector<float>> potential(image.rows, vector<float>(image.cols, 0));
        produce_receptive_field_sse(image, this->kernel, potential);

        // count how many times each neuron spiked on a particular potential
        vector<int> spikes_per_neuron = this->train_on_potential(potential);
        // find the neuron with the most spikes
        std::vector<int>::iterator max = max_element(spikes_per_neuron.begin(), spikes_per_neuron.end());

        // find the index of the neuron with the highest number of spikes
        return distance(spikes_per_neuron.begin(), max);
    }

    return -1;
}

/*
 * Destructor
 * frees the heap allocated 2d float array synapse before destroying the object
 */
Network::~Network(){
    for(int i = 0; i < this->out_dim; i++){
        free(this->synapse[i]);
    } free(this->synapse);
}
