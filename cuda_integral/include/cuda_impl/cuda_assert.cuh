//
// Created by fenix on 5/15/20.
//

#ifndef CUDA_INTEGRAL_CUDA_ASSERT_CUH
#define CUDA_INTEGRAL_CUDA_ASSERT_CUH

#include <cstdio>

#define gpuErrorCheck(ans); { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif //CUDA_INTEGRAL_CUDA_ASSERT_CUH
