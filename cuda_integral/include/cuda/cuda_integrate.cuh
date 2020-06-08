//
// Created by fenix on 5/14/20.
//

#ifndef CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
#define CUDA_INTEGRAL_CUDA_INTEGRATE_CUH

#include <cuda/cuda_assert.cuh>
#include <iostream>


#define COEF_NUM 5
#define MAX_THREAD_NUM 256

// steps describe the number of steps for etch x and y separately
__global__ void cuda_thread_integrate(const double start_x, const double end_x,
                                      double start_y, double end_y, double dxy, size_t steps_per_thread, double *res,
                                      const double *d_c, const double *d_a1, const double *d_a2) {

    double l_res = 0.0; // local result
    start_y += dxy * steps_per_thread * threadIdx.x;
    end_y = start_y + dxy * steps_per_thread;

    double x = start_x, y = start_y, diag;
    while (y < end_y) {
        while (x < end_x) {
            for (uint8_t i = 0; i < COEF_NUM; ++i) {
                diag = (x - d_a1[i]) * (x - d_a1[i]) + (y - d_a2[i]) * (y - d_a2[i]);
                l_res += d_c[i] * exp(-diag / static_cast<double >(M_PI)) * cos(static_cast<double >(M_PI) * diag);
            }
            x += dxy;
        }
        x = start_x;
        y += dxy;
    }
    res[threadIdx.x] = -l_res * dxy * dxy;

//    __shared__ double local_res[MAX_THREAD_NUM], tmp_res[MAX_THREAD_NUM], tmp_diag[MAX_THREAD_NUM]; // local result
    /*
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
                tmp_res[threadIdx.x] += c[i] * exp(-1 / M_PI * tmp_diag[threadIdx.x]) * cos(M_PI * tmp_diag[threadIdx.x]);
            }
            ///////////////////////////////////////////////////////////////////////////////
            local_res[threadIdx.x] -= tmp_res[threadIdx.x];
            x += dxy;
        }
        start_y += dxy;
    }
    res[threadIdx.x] = local_res[threadIdx.x] * dxy * dxy;
     */
}


double cuda_integrate(size_t steps, const ConfigFileOpt &config) {
    double res = 0.0;
//  dxy - the length of the side of one integration square
    double dxy = (config.get_y().second - config.get_y().first) / static_cast<double>(steps);
    size_t steps_per_thread = steps / config.get_flow_num();
    double *d_res, *d_c, *d_a1, *d_a2; //! Device

    gpuErrorCheck(cudaMalloc((void **) &d_res, config.get_flow_num() * sizeof(double)))\
 gpuErrorCheck(cudaMalloc((void **) &d_c, COEF_NUM * sizeof(double)))\
 gpuErrorCheck(cudaMalloc((void **) &d_a1, COEF_NUM * sizeof(double)))\
 gpuErrorCheck(cudaMalloc((void **) &d_a2, COEF_NUM * sizeof(double)))\

gpuErrorCheck(cudaMemcpy(d_c, &(config.get_c()[0]), COEF_NUM * sizeof(double), cudaMemcpyHostToDevice));\
gpuErrorCheck(cudaMemcpy(d_a1, &(config.get_a1()[0]), COEF_NUM * sizeof(double), cudaMemcpyHostToDevice));\
gpuErrorCheck(cudaMemcpy(d_a2, &(config.get_a2()[0]), COEF_NUM * sizeof(double), cudaMemcpyHostToDevice));\

    cuda_thread_integrate<<<1, config.get_flow_num()>>>(config.get_x().first, config.get_x().second,
                                                        config.get_y().first, config.get_y().second,
                                                        dxy, steps_per_thread, d_res, d_c, d_a1, d_a2);

    /////////////////////////////////// Finalize result ///////////////////////////////////
    double h_res[config.get_flow_num()]; // output buffer
    gpuErrorCheck(cudaMemcpy(h_res, d_res, config.get_flow_num() * sizeof(double), cudaMemcpyDeviceToHost));
    for (ptrdiff_t i = 0; i < config.get_flow_num(); ++i) {
        res += h_res[i];
    }

    gpuErrorCheck(cudaFree(d_res))\
 gpuErrorCheck(cudaFree(d_c))\
 gpuErrorCheck(cudaFree(d_a1))\
 gpuErrorCheck(cudaFree(d_a2))\
    return res;
}

#endif //CUDA_INTEGRAL_CUDA_INTEGRATE_CUH
