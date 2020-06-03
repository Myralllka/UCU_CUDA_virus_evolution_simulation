//
// Created by fenix on 5/23/20.
//

#include "objects/state_obj.h"

State State::NoneState = State{};

State States::normal = State{};
State States::immunity = State{};
State States::infected = State{};
State States::patient = State{};
State States::isolated = State{};
State States::dead = State{};
float States::crit_prob = 0.0f;
uint8_t States::incubation_time = 1;
const State *States::states_v[STATES_NUM] = {&States::normal, &States::immunity, &States::infected, &States::patient,
                                             &States::isolated, &States::dead};

State &State::operator()(int new_id, char new_repr, float new_prob, State &new_next) {
    id = new_id;
    repr = new_repr;
    prob = new_prob;
    next = std::ref(new_next);
    return *this;
}

bool State::operator==(const State &other) const {
    return (id & PURE_ID_MASK) == (other.id & PURE_ID_MASK);
}

bool State::operator!=(const State &other) const {
    return (id & PURE_ID_MASK) != (other.id & PURE_ID_MASK);
}

State::operator char() const {
    return repr;
}

bool State::mask_check(uint8_t mask) const {
    return id & mask;
}
