//
// Created by fenix on 5/14/20.
//

#ifndef CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
#define CUDA_INTEGRAL_CUDA_INTEGRATE_CUH

#define _USE_MATH_DEFINES

#include "../math/integration_args.h"
#include "cuda_assert.cuh"
#include <cmath>

#define COEF_NUM 5
//struct langerman_coefs_t {
//    double *c, *a1, *a2;
//};

// TODO: consider duplications!!!
extern __constant__ double c[COEF_NUM], a1[COEF_NUM], a2[COEF_NUM];
//extern __constant__ langerman_coefs_t coefs;

// "diagonal" calculation (for Langermann function simplification)
inline static double diag_len(const double &x, const double &y, unsigned long int i) {
    return (x - a1[i]) * (x - a1[i]) + (y - a2[i]) * (y - a2[i]);
}

// ########################### Langermann function ###########################
inline double langermann_f(const double x, const double y) {
    double res = 0.0;
    double diag;
    for (unsigned long int i = 0; i < COEF_NUM; ++i) {
        diag = diag_len(x, y, i);
        res += c[i] * exp(-1 / M_PI * diag) * cos(M_PI * diag);
    }
    return -res;
}

// steps describe the number of steps for etch x and y separately
template<typename func_T>
__global__ void cuda_thread_integrate(func_T func, const double start_x, const double end_x,
                                      double start_y, double end_y, double dxy, size_t steps, double *res) {
    double x, l_res = 0.0; // local result

    start_y = start_y + dxy * steps * threadIdx.x;
    end_y = start_y + dxy * steps * (threadIdx.x + 1);

    while (start_y < end_y) {
        x = start_x;
        while (x < end_x) {
            l_res += func(x, start_y);
            x += dxy;
        }
        start_y += dxy;
    }
    *res = l_res * dxy * dxy;
}


template<typename func_T>
double cuda_integrate(func_T func, size_t steps, const integration_args &int_ars) {
    double res = 0.0;
//  dxy - the length of the side of one integration square
    double dxy = sqrt((int_ars.end.x - int_ars.start.x) * (int_ars.end.y - int_ars.start.y) / steps);
    size_t steps_per_thread = (int_ars.end.y - int_ars.start.y) / (dxy * int_ars.flow_n) + 1;
    double *d_res; //! Device
    gpuErrorCheck(cudaMalloc((void **) &d_res, int_ars.flow_n * sizeof(double)));

    cuda_thread_integrate<<<1, int_ars.flow_n>>>(func, int_ars.start.x, int_ars.end.x,
                                                 int_ars.start.y, int_ars.end.y, dxy, steps_per_thread, d_res);

    /////////////////////////////////// Finalize result ///////////////////////////////////
    double h_res[int_ars.flow_n];
    gpuErrorCheck(cudaMemcpy(h_res, d_res, int_ars.flow_n * sizeof(double), cudaMemcpyDeviceToHost));
    for (ptrdiff_t i = 0; i < int_ars.flow_n; ++i) {
        res += h_res[i];
    }

    gpuErrorCheck(cudaFree(d_res));
    return res;
}

#endif //CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
