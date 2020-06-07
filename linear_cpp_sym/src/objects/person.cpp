//
// Created by fenix on 5/23/20.
//

#include <objects/field.h>

State Person::NoneState = State{};

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

bool Person::is_isolated() const {
    return state.get() == States::isolated;
}

bool Person::is_healthy() const {
    return state.get() == States::normal;
}

bool Person::is_infectious() const {
    return state.get() == States::infected or state.get() == States::patient;
//    return state.get().mask_check(STATE_INFECT);
};

void Person::become_infected() {
    state = States::infected;
}

void Person::become_dead() {
    next_state = States::dead;
}

char Person::get_repr() const {
    return static_cast<char>(state.get());
}

bool Person::try_infect() {
    if (is_healthy())
        next_state = random_state_choice(next_state.get().prob, States::normal, States::infected);
    return next_state.get() == States::infected;
}

uint8_t Person::evolute(size_t *isolation_places, std::unordered_map<State, size_t, StateHasher> &statistics) {
    switch (state.get().repr) {
        case NORMAL_STATE_CHAR:
            break;
        case IM_NORMAL_STATE_CHAR:
            next_state = States::immunity;
            break;
        case INFECTED_STATE_CHAR:
            next_state = random_state_choice(States::infected.prob, States::infected, States::patient);
            if (next_state.get() == States::patient) {
                ++statistics[States::patient];
                --statistics[States::infected];
            }
            break;
        case ISOLATED_STATE_CHAR:
            if (random_bool(States::patient_coef)) {
                ++(*isolation_places);
                --statistics[States::isolated];
                next_state = random_state_choice(state.get().prob, States::immunity, States::dead);
                next_state.get() == States::immunity ? ++statistics[States::immunity] : ++statistics[States::dead];
            } else {
                next_state = States::isolated;
            }
            break;
        case PATIENT_STATE_CHAR:
            if (random_bool(States::patient_coef)) {
                --statistics[States::patient];
                next_state = random_state_choice(state.get().prob, States::immunity, States::dead);
                next_state.get() == States::immunity ? ++statistics[States::immunity] : ++statistics[States::dead];
            } else if (*isolation_places) {
                --(*isolation_places);
                ++statistics[States::isolated];
                --statistics[States::patient];
                next_state = States::isolated;
            }
            break;
        case DEAD_STATE_CHAR:
            become_dead();
            break;
        default:
            std::cerr << "unexpected state" << std::endl;
    }
    if (state.get() == States::normal and next_state.get() == States::infected) {
        --statistics[States::normal];
        ++statistics[States::infected];
    }
    state = next_state;
    return state.get().id;
}
