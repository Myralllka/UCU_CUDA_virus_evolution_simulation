//
// Created by fenix on 5/23/20.
//

#include "objects/person.h"

State Person::NoneState = State{};

void Person::set_timer(uint8_t t) {
    state_timer = t;
}

State &Person::get_state() const {
    return state.get();
}

bool Person::is_alive() const {
    return state.get() != States::dead;
}

bool Person::is_patient() const {
    return state.get() == States::patient;
}

bool Person::is_infected() const {
    return state.get() == States::infected;
}

bool Person::is_healthy() const {
    return state.get() == States::normal;
}

bool Person::is_transmissible() const {
    return state.get().mask_check(STATE_INFECT);
};

bool Person::has_immunity() const {
    return state.get().mask_check(STATE_IMMUNITY);
}

void Person::become_healthy() {
    next_state = States::normal;
}

void Person::become_infected() {
    next_state = States::infected;
}

void Person::become_patient() {
    next_state = States::patient;
}

void Person::become_dead() {
    next_state = States::dead;
}

char Person::get_repr() const {
    return static_cast<char>(state.get());
}

bool Person::try_infect() {
    if (next_state.get() == States::normal) {
        next_state = random_state_choice(next_state.get().prob, next_state, States::infected);
        if (States::infected == next_state) {
            state_timer = States::incubation_time;
            return true;
        }
    }
    return false;
}

void Person::evolute(size_t *isolation_places) {
    if (state.get() == States::infected) {
        if (state_timer > 0)
            state_timer -= 1;
        else
            next_state = States::patient;
    } else if (state.get() == States::patient) {
        if (random_bool(States::crit_prob)) {
            if (!is_transmissible()) (*isolation_places)++;
            next_state = random_state_choice(state.get().prob, States::immunity, States::dead);
        } else if (*isolation_places && is_transmissible()) {
//            std::cout << "test" << std::endl;
            *isolation_places -= 1;
            next_state = States::isolated;
        }
    }
    state = next_state;
}
