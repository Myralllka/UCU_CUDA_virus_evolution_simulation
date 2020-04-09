//
// Created by fenix on 4/9/20.
//

#ifndef CUDA_SANDBOX_VEC_ADD_CUH
#define CUDA_SANDBOX_VEC_ADD_CUH

__global__ void vecAdd(double *a, double *b, double *c, int n);

void test_vec_add();

#endif //CUDA_SANDBOX_VEC_ADD_CUH
