//
// Created by fenix on 6/13/20.
//
#include <cuda_simulation.cuh>
#include <cuda_assert.cuh>

#define PRINT_DELAY_ITERS 1u
#define NAMED_OUTPUT
#define DEBUG
#define THREADS 16u

#ifdef DEBUG

#include <bitset>
#include <vector>

#endif // DEBUG


__global__ void statistics_test(Statistics *d_res_stats) {
    *d_res_stats = Statistics{555, 5555555555, 55, 5, 5, 5};
}


__global__ void
init_working_set(uint8_t *d_field, uint8_t *d_next_field, size_t field_side_len, curandState_t *d_rand_gen_arr,
                 size_t *d_isolation_places_arr, size_t isol_place, Statistics *d_res_stats) {
    // CUDA's random number library uses curandState_t to keep track of the seed value we will store
    // a random state for every thread
    curandState_t state;
    curand_init(clock64() - 1u,         /* the seed controls the sequence of random values that are produced */
                0, /* the sequence number is only important with multiple cores */
                0,      /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);

//     fills with IMMUNITY_ID by default
    fill_array_border(d_field, field_side_len);
    fill_array_border(d_next_field, field_side_len);
//     infect first cell
    size_t first_infected_coord = coord((curand(&state) % (field_side_len - 2u)) + 1u,
                                        (curand(&state) % (field_side_len - 2u)) + 1u,
                                        field_side_len);
    d_field[first_infected_coord] = INFECTED_ID;
    d_next_field[first_infected_coord] = INFECTED_ID;

    // init random generator for each thread
    for (size_t offset = 0; offset <= THREADS; ++offset)
        curand_init(clock64(),         /* the seed controls the sequence of random values that are produced */
                    offset, /* the sequence number is only important with multiple cores */
                    0u,      /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &d_rand_gen_arr[offset]);

    // distribute isol places for each thread
    // TODO: check if isol_place is decidable by THREADS
    const size_t isol_place_per_thread = isol_place / THREADS;
    for (size_t i = 0; i < THREADS; ++i)
        d_isolation_places_arr[i] = isol_place_per_thread;

    // init statistics
    *d_res_stats = Statistics{};
}

void cuda_simulation(ConfigFileOpt config) {
    size_t isolation_places = config.isol_place;
    const size_t field_side_len = config.field_size + 2u;

//  probabilities from one state to next state
//  healthy -> infected -> patient -> patient_critical(only one era) -> dead
//                                                                  -> immunity
    // TODO: load into CUDA_CONST_MEMORY
    const float probab_arr[NUMBER_OF_STATES] = {config.healthy_to_infected,         // healthy
                                                config.infected_to_patient,         // infected
                                                config.patient_coefficient,         // patient
                                                config.patient_to_dead,             // patient_crit
                                                FINAL_NEXT_STATE_PROBAB,            // dead
                                                FINAL_NEXT_STATE_PROBAB,            //
                                                FINAL_NEXT_STATE_PROBAB,            //
                                                FINAL_NEXT_STATE_PROBAB,            //
                                                FINAL_NEXT_STATE_PROBAB};           // immunity

    ///////////////////////// INIT WORKING FIELDS //////////////////////////

    uint8_t *d_field, *d_next_field;
    // TODO: move probab_arr to const memory
    float *d_probab_arr;
    curandState_t *d_rand_gen_arr;
    size_t *d_isolation_places_arr;
    Statistics *d_res_stats, res_stats{};

    gpuErrorCheck(cudaMalloc((void **) &d_field, field_side_len * field_side_len * sizeof(uint8_t)))//
    gpuErrorCheck(cudaMalloc((void **) &d_next_field, field_side_len * field_side_len * sizeof(uint8_t)))//
    gpuErrorCheck(cudaMalloc((void **) &d_probab_arr, NUMBER_OF_STATES * sizeof(float)))//
    gpuErrorCheck(cudaMalloc((void **) &d_rand_gen_arr, THREADS * sizeof(curandState_t)))//
    gpuErrorCheck(cudaMalloc((void **) &d_isolation_places_arr, THREADS * sizeof(size_t)))//
    gpuErrorCheck(cudaMalloc((void **) &d_res_stats, sizeof(Statistics)))//

    gpuErrorCheck(cudaMemcpy(d_probab_arr, probab_arr, NUMBER_OF_STATES * sizeof(float), cudaMemcpyHostToDevice))//
    gpuErrorCheck(cudaMemset(static_cast<void *>(d_field), 0, field_side_len * field_side_len * sizeof(uint8_t)))//
    gpuErrorCheck(cudaMemset(static_cast<void *>(d_next_field), 0, field_side_len * field_side_len * sizeof(uint8_t)))//

    init_working_set<<<1, 1>>>(d_field, d_next_field, field_side_len, d_rand_gen_arr, d_isolation_places_arr,
                               config.isol_place, d_res_stats);
    ///////////////////////// END INIT WORKING FIELDS //////////////////////

    // indicate witch d_field is current and witch next
    bool next = true;

#ifdef DEBUG
    std::vector<uint8_t> tmp_v(field_side_len * field_side_len);
    uint8_t *tmp_field = tmp_v.data();
//    Statistics tmp_s{};
//    float prob[NUMBER_OF_STATES];
#endif // DEBUG

    for (size_t i = 0u; i < config.num_of_eras / PRINT_DELAY_ITERS; ++i) { gpuErrorCheck(
                cudaMemcpy(&res_stats, d_res_stats, sizeof(Statistics), cudaMemcpyDeviceToHost))//
        // TODO: statistics calculation
        ///////////////////// OUTPUT OUTLAY ////////////////////
        // normal, immunity, infected, patient, isolated, dead;
#ifdef NAMED_OUTPUT
        std::cout << "immunity " << res_stats.immunity << " "
                  << "infected " << res_stats.infected << " "
                  << "patient " << res_stats.patient << " "
                  << "isolated " << res_stats.isolated << " "
                  << "dead " << res_stats.dead << std::endl;
#else
        std::cout << res_stats.immunity << " "
                  << res_stats.infected << " "
                  << res_stats.patient << " "
                  << res_stats.isolated << " "
                  << res_stats.dead << std::endl;
#endif // NAMED_OUTPUT

        if (res_stats.immunity + res_stats.dead == (field_side_len - 2u) * (field_side_len - 2u)) {
            // finish simulation after system stabilization
            return;
        }
        ///////////////////// OUTPUT OUTLAY END ////////////////

        //////////////////////////////////// DELAY LOOP  /////////////////////////////////////
        // TODO: assert thread divisible by 2
        dim3 worker_space(THREADS / 2u, THREADS / 2u);
//        dim3 worker_space(1, 1);
//        dim3 worker_space(32, 32);
//        for (size_t print_delay = 0u; print_delay < PRINT_DELAY_ITERS; ++print_delay) {
        if (next)
            sim_block_worker<<<1, worker_space>>>(d_field, d_next_field, field_side_len, d_probab_arr,
                                                  &isolation_places, d_rand_gen_arr, d_res_stats);
        else
            sim_block_worker<<<1, worker_space>>>(d_next_field, d_field, field_side_len, d_probab_arr,
                                                  &isolation_places, d_rand_gen_arr, d_res_stats);


        std::vector<uint8_t> v(field_side_len * field_side_len);
        uint8_t *h_field = v.data();

//        if (next) {
//            gpuErrorCheck(cudaMemcpy(h_field, d_field,
//                                     field_side_len * field_side_len * sizeof(uint8_t), cudaMemcpyDeviceToHost))//
//        } else {
//            gpuErrorCheck(cudaMemcpy(h_field, d_next_field,
//                                     field_side_len * field_side_len * sizeof(uint8_t), cudaMemcpyDeviceToHost))//
//        }


//        for (size_t row = 0; row < field_side_len; ++row) {
//            for (size_t col = 0; col < field_side_len; ++col)
//                std::cout << std::bitset<8>(h_field[row * field_side_len + col]) << " ";
//            std::cout << std::endl;
//        }
//
        next = !next;
//        }
        //////////////////////////////////// DELAY LOOP END //////////////////////////////////
    }gpuErrorCheck(cudaMemcpy(&res_stats, d_res_stats, sizeof(Statistics), cudaMemcpyDeviceToHost))//
    // TODO: statistics calculation
    ///////////////////// OUTPUT OUTLAY ////////////////////
    // normal, immunity, infected, patient, isolated, dead;
#ifdef NAMED_OUTPUT
    std::cout << "immunity " << res_stats.immunity << " "
              << "infected " << res_stats.infected << " "
              << "patient " << res_stats.patient << " "
              << "isolated " << res_stats.isolated << " "
              << "dead " << res_stats.dead << std::endl;
#else
    std::cout << res_stats.immunity << " "
                  << res_stats.infected << " "
                  << res_stats.patient << " "
                  << res_stats.isolated << " "
                  << res_stats.dead << std::endl;
#endif // NAMED_OUTPUT

    gpuErrorCheck(cudaFree(d_rand_gen_arr))//
    gpuErrorCheck(cudaFree(d_isolation_places_arr))//
    gpuErrorCheck(cudaFree(d_res_stats))//
    gpuErrorCheck(cudaFree(d_probab_arr))//
    gpuErrorCheck(cudaFree(d_next_field))//
    gpuErrorCheck(cudaFree(d_field))//
}

__global__ void sim_block_worker(uint8_t *d_field, uint8_t *d_next_field, size_t field_side_len,
                                 const float *probab_arr, size_t *isolation_places, curandState_t *d_rand_gen_arr,
                                 Statistics *d_res_stats) {
    // TODO: assert that field_side_len / blockDim.x is fully dividable
    const size_t working_set_side = (field_side_len - 2u) / blockDim.x;
    const uint thread_id = threadIdx.x + blockDim.x * threadIdx.y;

    __shared__ size_t stats_arr[
            (THREADS / 2) * (THREADS / 2) * NUMBER_OF_STATES]; // UNUSED_ID index is used for isolated count
    for (size_t i = thread_id * NUMBER_OF_STATES; i < (thread_id + 1) * NUMBER_OF_STATES; ++i)
        stats_arr[i] = 0;
//    if (thread_id == THREADS - 1) {
//        for (size_t i = 0; i < NUMBER_OF_STATES * THREADS; ++i)
//            stats_arr[i] = 0;
//    }

    uint8_t cell_st_id;
//    for (size_t row = 1u; row < field_side_len - 1u; ++row)
    size_t row, col;
    for (row = 1u + working_set_side * threadIdx.y; row < working_set_side * (threadIdx.y + 1); ++row)
//        for (size_t col = 1u; col < field_side_len - 1u; ++col) {
        for (col = 1u + working_set_side * threadIdx.x; col < 1u + working_set_side * (threadIdx.x + 1); ++col) {
            cell_st_id = d_field[coord(row, col, field_side_len)];

//            if (cell_st_id & FINAL_STATE_CHECK_MASK) {
//                d_next_field[coord(row, col, field_side_len)] = cell_st_id;
//                continue;
//            } else
            if (cell_st_id == HEALTHY_ID) {
                infect_cell(d_field[coord(row - 1u, col, field_side_len)], cell_st_id, probab_arr,
                            &(d_rand_gen_arr[thread_id]));
                infect_cell(d_field[coord(row, col - 1u, field_side_len)], cell_st_id, probab_arr,
                            &(d_rand_gen_arr[thread_id]));
                infect_cell(d_field[coord(row, col + 1u, field_side_len)], cell_st_id, probab_arr,
                            &(d_rand_gen_arr[thread_id]));
                infect_cell(d_field[coord(row + 1u, col, field_side_len)], cell_st_id, probab_arr,
                            &(d_rand_gen_arr[thread_id]));

                d_next_field[coord(row, col, field_side_len)] = cell_st_id;
                continue;
            }

            ///////////////////////////  ISOLATE IF POSSIBLE  ///////////////////////////
//            if (!(cell_st_id & ISOLATE_MASK))
//                if (cell_st_id == PATIENT_ID)
//                    if (*isolation_places) {
//                        --(*isolation_places);
//                        cell_st_id = ISOLATE_MASK | PATIENT_ID;
//                    }
            ///////////////////////////  ISOLATE IF POSSIBLE END  ///////////////////////

            ///////////////////////////  NEXT STATE  ///////////////////////
//            if (random_bool(probab_arr[cell_st_id & REMOVE_ISOL_MASK], &(d_rand_gen_arr[thread_id]))) {
//                ++cell_st_id;
//                if ((cell_st_id & FINAL_STATE_CHECK_MASK) && (cell_st_id & ISOLATE_MASK)) {
//                    ++(*isolation_places); // return an isolation place
//                    cell_st_id &= REMOVE_ISOL_MASK;
//                }
//            } else if ((cell_st_id & REMOVE_ISOL_MASK) == PATIENT_CRIT_ID) {
//                if (cell_st_id & ISOLATE_MASK)
//                    ++(*isolation_places); // return an isolation place
//                cell_st_id = IMMUNITY_ID;
//            }
            ///////////////////////////  NEXT STATE END  ///////////////////
//            d_next_field[coord(row, col, field_side_len)] = cell_st_id;

            /////////////////////////// STATISTICS GATHERING ///////////////////////
//          if (   is_isolated           )
            if (cell_st_id & ISOLATE_MASK)
                ++stats_arr[UNUSED_ID + thread_id * NUMBER_OF_STATES];
            else
                ++stats_arr[cell_st_id + thread_id * NUMBER_OF_STATES];
            /////////////////////////// STATISTICS GATHERING END ///////////////////
        }

    __syncthreads();
    if (thread_id == 0) {
        // TODO: fix THREADS definition
        for (uint thread_offset = 0; thread_offset < (THREADS / 2) * (THREADS / 2); ++thread_offset) {
            stats_arr[HEALTHY_ID] += stats_arr[HEALTHY_ID + thread_offset * NUMBER_OF_STATES];
            stats_arr[IMMUNITY_ID] += stats_arr[IMMUNITY_ID + thread_offset * NUMBER_OF_STATES];
            stats_arr[INFECTED_ID] += stats_arr[INFECTED_ID + thread_offset * NUMBER_OF_STATES];
            stats_arr[PATIENT_ID] += stats_arr[PATIENT_ID + thread_offset * NUMBER_OF_STATES];
            stats_arr[UNUSED_ID] += stats_arr[UNUSED_ID + thread_offset * NUMBER_OF_STATES];
            stats_arr[DEAD_ID] += stats_arr[DEAD_ID + thread_offset * NUMBER_OF_STATES];
        }
        *d_res_stats = Statistics{stats_arr[HEALTHY_ID],
                                  stats_arr[IMMUNITY_ID],
                                  stats_arr[INFECTED_ID] + stats_arr[PATIENT_ID] + stats_arr[PATIENT_CRIT_ID],
                                  stats_arr[PATIENT_ID] + stats_arr[PATIENT_CRIT_ID] + stats_arr[UNUSED_ID],
                                  stats_arr[UNUSED_ID],
                                  stats_arr[DEAD_ID]
        };
    }
}

