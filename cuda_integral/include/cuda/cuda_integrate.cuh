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
#define MAX_THREAD_NUM 256

// TODO: consider duplications!!!
extern __constant__ double c[COEF_NUM], a1[COEF_NUM], a2[COEF_NUM];

// steps describe the number of steps for etch x and y separately
__global__ void cuda_thread_integrate(const double start_x, const double end_x,
                                      double start_y, double end_y, double dxy, size_t steps, double *res) {

    __shared__ double local_res[MAX_THREAD_NUM], tmp_res[MAX_THREAD_NUM], tmp_diag[MAX_THREAD_NUM]; // local result
    double x;
    start_y = start_y + dxy * steps * threadIdx.x;
    end_y = start_y + dxy * steps * (threadIdx.x + 1);

    while (start_y < end_y) {
        x = start_x;
        while (x < end_x) {
            ////////////////////////// Langerman Function inline //////////////////////////
            tmp_res[threadIdx.x] = 0.0;
            for (unsigned long int i = 0; i < COEF_NUM; ++i) {
                tmp_diag[threadIdx.x] = (x - a1[i]) * (x - a1[i]) + (start_y - a2[i]) * (start_y - a2[i]);
                tmp_res[threadIdx.x] +=
                        c[i] * exp(-1 / M_PI * tmp_diag[threadIdx.x]) * cos(M_PI * tmp_diag[threadIdx.x]);
            }
            ///////////////////////////////////////////////////////////////////////////////
            local_res[threadIdx.x] -= tmp_res[threadIdx.x];
            x += dxy;
        }
        start_y += dxy;
    }
    res[threadIdx.x] = local_res[threadIdx.x] * dxy * dxy;
}


double cuda_integrate(size_t steps, const integration_args &int_ars) {
    double res = 0.0;
//  dxy - the length of the side of one integration square
    double dxy = sqrt((int_ars.end.x - int_ars.start.x) * (int_ars.end.y - int_ars.start.y) / steps);
    size_t steps_per_thread = (int_ars.end.y - int_ars.start.y) / (dxy * int_ars.flow_n) + 1;
    double *d_res; //! Device
    gpuErrorCheck(cudaMalloc((void **) &d_res, int_ars.flow_n * sizeof(double)));

    cuda_thread_integrate<<<1, int_ars.flow_n>>>(int_ars.start.x, int_ars.end.x,
                                                 int_ars.start.y, int_ars.end.y, dxy, steps_per_thread, d_res);

    /////////////////////////////////// Finalize result ///////////////////////////////////
    double h_res[int_ars.flow_n]; // output buffer
    gpuErrorCheck(cudaMemcpy(h_res, d_res, int_ars.flow_n * sizeof(double), cudaMemcpyDeviceToHost));
    for (ptrdiff_t i = 0; i < int_ars.flow_n; ++i) {
        res += h_res[i];
    }

    gpuErrorCheck(cudaFree(d_res));
    return res;
}

#endif //CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
