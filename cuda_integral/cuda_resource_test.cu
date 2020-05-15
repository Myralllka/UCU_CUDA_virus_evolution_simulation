#include <iostream>
#include "include/option_parser/ConfigFileOpt.h"
#include "include/math/integration_args.h"

#define COEF_NUM 5
#define gpuErrorCheck(ans); { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline static auto get_int_args_from_conf(const ConfigFileOpt &config) {
    return integration_args{
            point{config.get_x().first, config.get_y().first},
            point{config.get_x().second, config.get_y().second},
            f_params{
                    config.get_m(),
                    config.get_c(),
                    config.get_a1(),
                    config.get_a2()
            },
            config.get_flow_num()
    };
}


__constant__ double c[COEF_NUM], a1[COEF_NUM], a2[COEF_NUM];

int main(int argc, char *argv[]) {
//  //////////////////////////// Program Parameter Parsing ////////////////////////////
    std::string file_name = "execution.conf";

//  ////////////////////////////    Config File Parsing    ////////////////////////////
    ConfigFileOpt config{};
    try {
        config.parse(file_name);
    } catch (std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 3;
    }

    //  ////////////////////////////   Integration Initiation   ////////////////////////////
    const integration_args int_args = get_int_args_from_conf(config);


    double *d_a1;
    gpuErrorCheck(cudaMalloc(&d_a1,sizeof(double) * COEF_NUM))

    // Copy host vectors to device
    gpuErrorCheck(cudaMemcpy(d_a1, &config.get_a1()[0], sizeof(double) * COEF_NUM, cudaMemcpyHostToDevice))
//    gpuErrorCheck(cudaMemcpyToSymbol(&a1, &config.get_a1()[0], sizeof(double) * COEF_NUM, cudaMemcpyHostToDevice));

    double a1_out[COEF_NUM];gpuErrorCheck(
            cudaMemcpy(a1_out, d_a1, sizeof(double) * COEF_NUM, cudaMemcpyDeviceToHost));


    for (int i = 0; i < COEF_NUM; ++i)
        std::cout << a1_out[i] << "  ==  " << config.get_a1()[i] << std::endl;

    return 0;
}
