//
// Created by botsula on 14.05.20.
//
#include <stdio.h>
#include <iostream>
#include "state_obj.h"
#include "state_functions.h"


class Person {
private:
    static State NoneState;
    std::reference_wrapper<State> state = NoneState, next_state = NoneState;

public:
    Person() = default;

    Person(State &set_state, State &nextState) : state(std::ref(set_state)), next_state(std::ref(nextState)) {}

    [[nodiscard]] State &get_state() const;

    [[nodiscard]] bool is_alive() const;

    [[nodiscard]] bool is_patient() const;

    [[nodiscard]] bool is_infected() const;

    [[nodiscard]] bool is_healthy() const;

    [[nodiscard]] bool is_infectious() const;

    [[nodiscard]] bool is_isolated() const;

    [[nodiscard]] bool has_immunity() const;

    void try_infect();

    void become_healthy();

    void become_isolated();

    void become_infected();

    void become_patient();

    void become_dead();

    [[nodiscard]] char get_repr() const;

    uint8_t evolute(size_t *isolation_places, std::unordered_map<State, size_t, StateHasher> &statistics) ;
};
