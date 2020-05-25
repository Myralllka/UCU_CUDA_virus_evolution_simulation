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
    uint8_t state_timer = 0;

public:
    Person() = default;

    Person(State &set_state, State &nextState) : state(std::ref(set_state)), next_state(std::ref(nextState)) {}

    void set_timer(uint8_t t);

    [[nodiscard]] State &get_state() const;

    [[nodiscard]] bool is_alive() const;

    [[nodiscard]] bool is_patient() const;

    [[nodiscard]] bool is_infected() const;

    [[nodiscard]] bool is_healthy() const;

    [[nodiscard]] bool is_transmissible() const;

    [[nodiscard]] bool has_immunity() const;

    bool try_infect();

    void become_healthy();

    void become_infected();

    void become_patient();

    void become_dead();

    char get_repr() const;

    void evolute(size_t *isolation_places);
};
