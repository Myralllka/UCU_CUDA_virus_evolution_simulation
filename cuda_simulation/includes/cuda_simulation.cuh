//
// Created by fenix on 6/13/20.
//

#ifndef CUDA_SIMULATION_CUDA_SIMULATION_CUH
#define CUDA_SIMULATION_CUDA_SIMULATION_CUH

#include <file_interface/conf_parser.h>
#include <curand_kernel.h>
#include "simul_helpers.cuh"


void cuda_simulation(const ConfigFileOpt &config);

__global__ void sim_block_worker(const uint8_t *d_field, uint8_t *d_next_field, size_t field_side_len,
                                 const float *probab_arr, size_t *d_isolation_places_arr, curandState_t *d_rand_gen_arr,
                                 Statistics *d_res_stats);

#endif //CUDA_SIMULATION_CUDA_SIMULATION_CUH
