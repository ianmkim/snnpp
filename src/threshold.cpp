#include <vector>

#include "params.hpp"
#include "threshold.hpp"

using namespace std;

float threshold(vector<vector<float>> &train){
    int train_len = train.at(0).size();
    int thresh = 0;

    for(int i = 0; i < train_len; i++){
        int col_sum = 0;
        for(int j = 0; j < train.size(); j++)
            col_sum +=  train.at(j).at(i);
        if(col_sum > thresh)
            thresh = col_sum;
    }

    return (thresh/3) * Params::scale;
}