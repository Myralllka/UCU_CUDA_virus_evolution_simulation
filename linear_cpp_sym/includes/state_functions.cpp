//
// Created by botsula on 14.05.20.
//

#include <iostream>
#include "state.h"
#include <time.h>
#include <random>

//  0 - normal state.       .    ->  1/3 INFECTED 1
//  1 - infected state.     *    ->  1   PATIENT  2
//  2 - patient state.      O    ->  1/3 DEAD     3
//  3 - dead state.        ' '   ->  2/3 NORMAL   0

State create_state(int state_id) {
    if (state_id == 0) {
        return State(state_id, ".", 1, 0.3);
    } else if (state_id == 1) {
        return State(state_id, "*", 2, 1);
    } else if (state_id == 2) {
        return State(state_id, "0", 3, 0.3);
    } else if (state_id == 3) {
        return State(state_id, " ", 0, 0.6);
    }
};

void set_struct(States &cur_struct) {
    cur_struct.normal = 0;
    cur_struct.infected = 1;
    cur_struct.patient = 2;
    cur_struct.dead = 3;

    cur_struct.get_normal = 0;
    cur_struct.get_infected = 1;
    cur_struct.get_patient = 2;
    cur_struct.get_dead = 3;
}


State random_state_choice(double worst_weight, int better_state, int worst_state) {
    srand((unsigned) time(NULL));
    float r = (float) rand() / RAND_MAX;

    if (r <= worst_weight){
        std::cout << "REPR: " << worst_state;
        return create_state(worst_state);
    } else {
        return create_state(better_state);
    }
}
