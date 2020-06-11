//
// Created by fenix on 6/10/20.
//

#ifndef LINEAR_CPP_SYM_SIMUL_HELPERS_H
#define LINEAR_CPP_SYM_SIMUL_HELPERS_H

#include <cstddef>
#include "matrix/m_matrix.h"

#define FINAL_NEXT_STATE_PROBAB     0
#define NUMBER_OF_STATES            6

// STATES
//                                0000'0000
#define HEALTHY_ID          0
//                                0000'0001
#define INFECTED_ID         1
//                                0000'0010
#define PATIENT_ID          2
//                                0000'0011
#define PATIENT_CRIT_ID     3
//                                0000'0100
#define DEAD_ID             4
//                                0000'1000
#define IMMUNITY_ID         8

// usage (state_id & INFECTED_CHECK_MASK) -- check if state infect healthy state
#define INFECTED_CHECK_MASK       0000'0011

// probabilities from one state to next
// healthy -> infected -> patient -> patient_critical(only one era) -> dead
//                                                                  -> immunity
extern float probab_arr[NUMBER_OF_STATES];


struct Statistics {
    size_t healthy = 0, immunity = 0, infected = 0, patient = 0, isolated = 0, dead = 0;
};

Statistics get_statistics(const m_matrix<uint8_t> &field);

inline bool random_bool(float prob) {
    return rand() / static_cast<float>(RAND_MAX) < prob;
}

inline void infect_cell(const uint8_t &cell, uint8_t &res_cell) {
    if (res_cell != INFECTED_ID && cell == HEALTHY_ID)
        if (random_bool(probab_arr[HEALTHY_ID]))
            res_cell = INFECTED_ID;
}

void change_the_era(const m_matrix<uint8_t> &field, m_matrix<uint8_t> &next_field);

// fill square border with value
inline void fill_array_border(uint8_t *arr, size_t side_size, uint8_t val = IMMUNITY_ID) {
    // upper and lover side separately to effectively use cache
    for (size_t col = 0; col < side_size; ++col)
        arr[col] = val;
    for (size_t row = 1; row < side_size - 1; ++row) {
        arr[row * side_size] = val;
        arr[row * side_size + side_size - 1] = val;
    }
    for (size_t col = 0; col < side_size; ++col)
        arr[side_size * (side_size - 1) + col] = val;
}


#endif //LINEAR_CPP_SYM_SIMUL_HELPERS_H
