//
// Created by fenix on 5/23/20.
//

#include "../../includes/objects/state_obj.h"

State State::NoneState = State{};

State States::normal = State{};
State States::infected = State{};
State States::patient = State{};
State States::dead = State{};

State &State::operator()(int new_id, char new_repr, float new_prob, State &new_next) {
    id = new_id;
    repr = new_repr;
    prob = new_prob;
    next = std::ref(new_next);
    return *this;
}

bool State::operator==(const State &other) const {
    return id == other.id;
}

bool State::operator!=(const State &other) const {
    return id != other.id;
}

State::operator char() const {
    return repr;
}
