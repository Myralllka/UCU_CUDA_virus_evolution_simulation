//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_STATE_FUNCTIONS_H
#define LINEAR_CPP_SYM_STATE_FUNCTIONS_H
#include "state.h"

State create_state(int state_id);

void set_struct(States &cur_struct);

State random_state_choice(double worst_weight, int worst_state, int better_state);
#endif //LINEAR_CPP_SYM_STATE_FUNCTIONS_H
