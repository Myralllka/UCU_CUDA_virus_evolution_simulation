#include "file_interface/conf_parser.h"
#include <cuda_simulation.cuh>

//#define TEST_SPEED

#ifdef TEST_SPEED
#include <speed_tester.h>
#endif // TEST_SPEED

ConfigFileOpt parse_conf(int argc, char *argv[]);


int main(int argc, char *argv[]) {
#ifdef TEST_SPEED
    auto start = get_current_time_fenced();
#endif // TEST_SPEED

    ConfigFileOpt config = parse_conf(argc, argv);
    cuda_simulation(config);

#ifdef TEST_SPEED
    auto finish = get_current_time_fenced();
    std::cout << "Total in s  :\t" << to_s(finish - start) << std::endl;
    std::cout << "Total in ms :\t" << to_ms(finish - start) << std::endl;
#endif // TEST_SPEED
    return 0;
}

ConfigFileOpt parse_conf(int argc, char *argv[]) {
    //  ##################### Program Parameter Parsing ######################
    std::string filename = "simulation.conf";
    if (argc == 2) {
        filename = argv[1];
    } else if (argc > 2) {
        std::cerr << "Too many arguments. Usage: \n"
                     "\tprogram [config-filename]\n" << std::endl;
        exit(1);
    }

    //  #####################    Config File Parsing    ######################
    ConfigFileOpt config{};
    try {
        config.parse(filename);
    } catch (std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        exit(3);
    }
    return config;
}
