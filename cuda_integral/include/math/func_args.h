//
// Created by fenix on 3/1/20.
//

#ifndef LAB_2_PARALLEL_INTEGRATION_FUNC_ARGS_H
#define LAB_2_PARALLEL_INTEGRATION_FUNC_ARGS_H

#include <vector>

struct point {
    double x;
    double y;
};

struct f_params {
    const size_t m;
    const std::vector<double> c;
    const std::vector<double> a1;
    const std::vector<double> a2;
};

#endif //LAB_2_PARALLEL_INTEGRATION_FUNC_ARGS_H
