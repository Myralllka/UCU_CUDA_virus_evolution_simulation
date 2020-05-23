//
// Created by fenix on 5/14/20.
//

#ifndef LINEAR_CPP_SYM_STATE_OBJ_H
#define LINEAR_CPP_SYM_STATE_OBJ_H

#include <cinttypes>
#include <functional>

struct State {
    uint8_t id = -1;
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
};

struct States {
    static State normal, infected, patient, dead;
};

struct Statistics {
    size_t normal, infected, patient, dead;
};

#endif //LINEAR_CPP_SYM_STATE_OBJ_H
