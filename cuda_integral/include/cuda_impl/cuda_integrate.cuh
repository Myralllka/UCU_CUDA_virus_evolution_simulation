//
// Created by fenix on 5/14/20.
//

#ifndef CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
#define CUDA_INTEGRAL_CUDA_INTEGRATE_CUH

#include <cuda_impl/cuda_assert.cuh>
#include <option_parser/ConfigFileOpt.h>
#include <iostream>


#define COEF_NUM 5
#define MAX_THREAD_NUM 256

// steps describe the number of steps for etch x and y separately
__global__ void cuda_thread_integrate(double start_x, double end_x,
                                      double start_y, double end_y, double dxy, size_t steps_per_thread, double *res,
                                      const double *d_c, const double *d_a1, const double *d_a2);


double cuda_integrate(size_t steps, const ConfigFileOpt &config);

#endif //CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
