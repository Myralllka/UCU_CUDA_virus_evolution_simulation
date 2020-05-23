//const
// Created by fenix on 2/25/20.
//

#include "../../include/math/langermann_f.h"
#include <cmath>


// "diagonal" calculation (for Langermann function simplification)
inline static double diag_len(const point &p, const f_params &conf, size_t i) {
    return (p.x - conf.a1[i]) * (p.x - conf.a1[i]) + (p.y - conf.a2[i]) * (p.y - conf.a2[i]);
}

// ########################### Langermann function ###########################
double langermann_f(const point &p, const f_params &conf) {
    double res = 0.0;
    double diag;
    for (size_t i = 0; i < conf.m; ++i) {
        diag = diag_len(p, conf, i);
        res += conf.c[i] * exp(-1 / M_PI * diag) * cos(M_PI * diag);
    }
    return -res;
}