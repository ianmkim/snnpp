#include "doctest/doctest.h"

#include "learning.hpp"
#include "params.hpp"
#include "utils.hpp"

using namespace std;

TEST_CASE("Testing STDP reinforcement learning curve"){
    CHECK(sameFloat(reinforcement_learning(-20) * Params::sigma, 0.0005494691666620253));
    CHECK(sameFloat(reinforcement_learning(1)*Params::sigma, -0.07059975220676765));
}

TEST_CASE("Testing STDP weight update rule"){
    CHECK(sameFloat(stdp_update(10, 20), -7.0));
    CHECK(sameFloat(stdp_update(1, -50), 1.99999));
}