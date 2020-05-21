//
// Created by botsula on 14.05.20.
//
#include <stdio.h>
#include <iostream>
#include "state.h"
#include "state_functions.h"
#include "constants.h"


class Person {

public:
    State state, next_state;
    int static_state_timer;

    Person(State set_state, State nextState) : state(set_state), next_state(nextState) {
        static_state_timer = 0;
    }

    void set_static_timer(int t){
        static_state_timer = t;
    }

    State get_state() {
        return state;
    }

    bool is_alive() {
        return state.get_id() != 3;
    }

    bool is_patient() {
        return state.get_id() == 2;
    }

    bool is_infected() {
        return state.get_id() == 1;
    }

    bool is_healthy() {
        return state.get_id() == 0;
    }

    bool become_infected(States states_struct) {
        if (next_state.get_id() == states_struct.normal) {
            State next_state = random_state_choice(states_struct.get_infected,
                                                   state.get_next_state_id(), states_struct.infected);
            if (states_struct.infected == next_state.get_id()) {
                static_state_timer = INCUBATION_TIME;
                return true;
            }
        }
        return false;
    }

    void become_patient(States *states_struct) {
        next_state = create_state(states_struct->patient);
    }

    void become_dead(States *states_struct) {
        next_state = create_state(states_struct->dead);
    }

    void become_healthy(States *states_struct) {
        next_state = create_state(states_struct->normal);
    }

//    @property
//    def probability_become_patient(self):
//        return self._state[States.PATIENT]
//
//    @property
//    def probability_become_dead(self):
//        return self._state[States.DEAD]
//
//    @property
//    def probability_become_infected(self):
//        return self._state[States.INFECTED]

    void evolute(States states_struct) {
        if (state.get_id() == states_struct.infected) {
            if (static_state_timer > 0) {
                static_state_timer -= 1;
            } else next_state = create_state(states_struct.patient);
        } else if (state.get_id() == states_struct.patient) {
            next_state = random_state_choice(states_struct.get_dead,
                                             states_struct.normal, states_struct.dead);
        }
        state = next_state;
    }

    std::string get_repr() {
        return state.get_repr();
    }
};