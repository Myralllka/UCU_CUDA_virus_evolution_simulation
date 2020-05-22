//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_STATE_FUNCTIONS_H
#define LINEAR_CPP_SYM_STATE_FUNCTIONS_H

#include "objects/state_obj.h"
#include <ctime>
#include <random>

//State create_state(int state_id);

//void set_struct(States &cur_struct);

// TODO: need upgrade (set seed in main)
inline State &random_state_choice(float worst_weight, State &better_state, State &worst_state) {
    srand(static_cast<unsigned>(time(NULL)));
    return (rand() / static_cast<float>(RAND_MAX) <= worst_weight) ? worst_state : better_state;
}

#endif //LINEAR_CPP_SYM_STATE_FUNCTIONS_H

