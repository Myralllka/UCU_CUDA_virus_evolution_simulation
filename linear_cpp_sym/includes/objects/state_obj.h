//
// Created by fenix on 5/14/20.
//

#ifndef LINEAR_CPP_SYM_STATE_OBJ_H
#define LINEAR_CPP_SYM_STATE_OBJ_H

#include <cinttypes>
#include <functional>

// props bits
#define STATE_INFECT    0b1000'0000
#define STATE_IMMUNITY  0b0100'0000

// actual IDs
#define NORMAL_STATE_ID 0b0000'0001
#define IM_NORMAL_STATE_ID 0b0000'0100 | STATE_IMMUNITY
#define INFECTED_STATE_ID 0b0000'0010 | STATE_INFECT
#define PATIENT_STATE_ID 0b0000'0011 | STATE_INFECT
#define ISOLATED_STATE_ID 0b0000'0011
#define DEAD_STATE_ID 0b0000'0000

// check status
#define OK_STATE_ID 0b1111'1111

// masks
#define PURE_ID_MASK 0b0000'0111


struct State {
    // 0b 11 1111 111
    //    |             - infecting ability
    //     |            - immunity to the virus
    //            |||   - state identifier
    //       ||||       - this one is reserved

    uint8_t id = 0b1111'1111;
    char repr = '\0';
    float prob = 0.0f;

    static State NoneState;
    std::reference_wrapper<State> next = std::ref(NoneState);

    State() = default;

    explicit State(int new_id, char new_repr, float new_prob, State &new_next)
            : id(new_id), repr(new_repr), prob(new_prob), next(new_next) {}

    State &operator()(int new_id, char new_repr, float new_prob, State &new_next);

    [[nodiscard]] bool operator==(const State &other) const;

    [[nodiscard]] bool operator!=(const State &other) const;

    // TODO: check if valid
    [[nodiscard]] operator char() const;

    [[nodiscard]] bool mask_check(uint8_t mask) const;
};

struct States {
    static State normal, immunity, infected, patient, isolated, dead;
    static float crit_prob;
    static uint8_t incubation_time;
};

struct Statistics {
    size_t normal, immunity, infected, patient, isolated, dead;
};

#endif //LINEAR_CPP_SYM_STATE_OBJ_H
