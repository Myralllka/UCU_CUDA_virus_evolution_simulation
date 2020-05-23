//
// Created by fenix on 3/1/20.
//

#ifndef LAB_2_PARALLEL_INTEGRATION_INTEGRATION_ARGS_H
#define LAB_2_PARALLEL_INTEGRATION_INTEGRATION_ARGS_H

#include "func_args.h"
#include <cstddef>

struct integration_args {
    point start = point{0.0, 0.0};
    point end = point{0.0, 0.0};
    const f_params conf{};
    const uint16_t flow_n = 1;
    double dx = 0.0, dy = 0.0;
//    const size_t steps;
};

#endif //LAB_2_PARALLEL_INTEGRATION_INTEGRATION_ARGS_H
