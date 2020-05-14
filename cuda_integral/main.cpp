#include <iostream>
#include "include/option_parser/ConfigFileOpt.h"
#include "include/speed_tester.h"
#include "include/math/manual_integration.h"
#include "include/math/langermann_f.h"
#include <cmath>
#include <chrono>


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


int main(int argc, char *argv[]) {
//  ##################### Program Parameter Parsing ######################
    std::string file_name = "../execution.conf";
    if (argc == 2) {
        file_name = argv[1];
    }
    if (argc > 2) {
        std::cerr << "Too many arguments. Usage: \n"
                     "<program>\n"
                     "or\n"
                     "<program> <config-filename>\n" << std::endl;
        return 1;
    }

//  #####################    Config File Parsing    ######################
    ConfigFileOpt config{};
    try {
        config.parse(file_name);
    } catch (std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 3;
    }

//  #####################   Integration Initiation   ######################
    size_t steps = config.get_init_steps();
    const integration_args int_args = get_int_args_from_conf(config);

    auto before = get_current_time_fenced();
    double cur_res = integrate(langermann_f, steps, int_args);

    double prev_res = cur_res;
    bool to_continue = true;
    double abs_err = -1; // Just guard value
    double rel_err = -1; // Just guard value

//  #####################   Main Calculation Cycle   ######################
// #define PRINT_INTERMEDIATE_STEPS
    while (to_continue) {

#ifdef PRINT_INTERMEDIATE_STEPS
        std::cout << cur_res << " : " << steps << " steps" << std::endl;
#endif

        prev_res = cur_res;
        steps *= 2;
        cur_res = integrate(langermann_f, steps, int_args);
        abs_err = fabs(cur_res - prev_res);
        rel_err = fabs((cur_res - prev_res) / cur_res);

#ifdef PRINT_INTERMEDIATE_STEPS
        std::cout << '\t' << "Abs err : rel err " << abs_err << " : " << rel_err << std::endl;
#endif

        to_continue = (abs_err > config.get_abs_pars());
        to_continue = to_continue && (rel_err > config.get_rel_pres());
        to_continue = to_continue && (steps < config.get_max_steps());
    }

    auto time_to_calculate = get_current_time_fenced() - before;

    
//  #####################   Program Output Block   ######################
//    std::cout << "Steps: " << steps << std::endl;
//    std::cout << "Flows: " << config.get_flow_num() << std::endl;
    std::cout << "Result = " << cur_res << std::endl;
    std::cout << "Absolute_error = " << abs_err << std::endl;
    std::cout << "Relative_error = " << rel_err << std::endl;
    std::cout << "Time = " << to_us(time_to_calculate) << std::endl;

    return 0;
}
