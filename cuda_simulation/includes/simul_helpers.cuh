//
// Created by fenix on 6/10/20.
//

#ifndef LINEAR_CPP_SYM_SIMUL_HELPERS_H
#define LINEAR_CPP_SYM_SIMUL_HELPERS_H

#include <cstddef>
#include <cinttypes>
#include <curand_kernel.h>

#define FINAL_NEXT_STATE_PROBAB     0.f
// maximal state id + 1
#define NUMBER_OF_STATES            9u

// STATES
//                                0000'0000
#define HEALTHY_ID          0u
//                                0000'0001
#define INFECTED_ID         1u
//                                0000'0010
#define PATIENT_ID          2u
//                                0000'0011
#define PATIENT_CRIT_ID     3u
//                                0000'0100
#define DEAD_ID             4u
//                                0000'1000
#define IMMUNITY_ID         8u
// for temporary usage
#define UNUSED_ID           5u

// usage (state_id & INFECTED_CHECK_MASK) -- check if state infect healthy state
#define INFECTED_CHECK_MASK             0b0000'0011u

// usage (state_id & FINAL_STATE_CHECK_MASK) -- check if final state
#define FINAL_STATE_CHECK_MASK          0b0000'1100u

// usage (state_id | ISOLATE_MASK) -- isolate a state
#define ISOLATE_MASK                    0b0001'0000u

// usage (state_id & ISOLATE_MASK) -- get clean state id
#define REMOVE_ISOL_MASK                0b1110'1111u


struct Statistics {
    size_t healthy = 0, immunity = 0, infected = 0, patient = 0, isolated = 0, dead = 0;
};


__device__ inline bool random_bool(float prob, curandState_t *state_p) {
    return static_cast<float>(curand(state_p)) / static_cast<float>(RAND_MAX) < prob;
}

__device__ inline void
infect_cell(const uint8_t &cell, uint8_t &infect_cell, const float *probab_arr, curandState_t *state_p) {
    if (!(cell & ISOLATE_MASK) && (cell & INFECTED_CHECK_MASK))
        if (random_bool(probab_arr[HEALTHY_ID], state_p))
            infect_cell = INFECTED_ID;
}

// coordinate in 2d square array
__device__ __forceinline__ size_t coord(const size_t &row, const size_t &col, size_t field_side_len) {
    return field_side_len * row + col;
}


__device__ inline Statistics get_statistics(const uint8_t *field, size_t field_side_len) {
    size_t res[NUMBER_OF_STATES]; // UNUSED_ID index is used for isolated count
    for (auto &re : res)
        re = 0u;

    uint8_t tmp_cell;
    for (size_t row = 1u; row < field_side_len - 1u; ++row)
        for (size_t col = 1u; col < field_side_len - 1u; ++col) {
            tmp_cell = field[coord(row, col, field_side_len)];
//          if (   is_isolated         )
            if (tmp_cell & ISOLATE_MASK)
                ++res[UNUSED_ID];
            else
                ++res[tmp_cell];
        }

    return Statistics{res[HEALTHY_ID],
                      res[IMMUNITY_ID],
                      res[INFECTED_ID] + res[PATIENT_ID] + res[PATIENT_CRIT_ID],
                      res[PATIENT_ID] + res[PATIENT_CRIT_ID] + res[UNUSED_ID],
                      res[UNUSED_ID],
                      res[DEAD_ID]
    };
}

// fill square border with value
__device__ inline void fill_array_border(uint8_t *arr, size_t side_size, uint8_t val = IMMUNITY_ID) {
    // upper and lover side separately to effectively use cache
    for (size_t col = 0u; col < side_size; ++col)
        arr[col] = val;
    for (size_t row = 1u; row < side_size - 1u; ++row) {
        arr[row * side_size] = val;
        arr[row * side_size + side_size - 1u] = val;
    }
    for (size_t col = 0u; col < side_size; ++col)
        arr[side_size * (side_size - 1u) + col] = val;
}


#endif //LINEAR_CPP_SYM_SIMUL_HELPERS_H
