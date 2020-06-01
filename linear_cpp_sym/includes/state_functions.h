//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_STATE_FUNCTIONS_H
#define LINEAR_CPP_SYM_STATE_FUNCTIONS_H

#include <random>
#include <ctime>
#include <iostream>
#include "objects/state_obj.h"


// TODO: need upgrade (set seed in main)
// TODO: use advanced random instead of c-random
inline State &random_state_choice(float worst_weight, State &better_state, State &worst_state) {
    return (rand() / static_cast<float>(RAND_MAX) <= worst_weight) ? worst_state : better_state;
}

inline bool random_bool(float prob) {
    return rand() / static_cast<float>(RAND_MAX) <= prob;
}

#endif //LINEAR_CPP_SYM_STATE_FUNCTIONS_H

