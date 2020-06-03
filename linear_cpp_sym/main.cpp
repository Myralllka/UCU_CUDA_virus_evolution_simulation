#include "objects/state_obj.h"
#include "objects/field.h"
#include "file_interface/conf_parser.h"

int main(int argc, char *argv[]) {
    //  ##################### Program Parameter Parsing ######################
    std::string filename = "simulation.conf";
    if (argc == 2) {
        filename = argv[1];
    } else if (argc > 2) {
        std::cerr << "Too many arguments. Usage: \n"
                     "\tprogram [config-filename]\n" << std::endl;
        return 1;
    }

    //  #####################    Config File Parsing    ######################
    ConfigFileOpt config{};
    try {
        config.parse(filename);
    } catch (std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 3;
    }

//  0 - normal state.       .    ->  1/3 INFECTED 1
//  1 - infected state.     *    ->  1   PATIENT  2
//  2 - patient state.      O    ->  1/3 DEAD     3
//  3 - dead state.        ' '   ->  2/3 NORMAL   0

    States::normal(NORMAL_STATE_ID, '.', config.norm_to_inf, States::infected);
    States::immunity(IM_NORMAL_STATE_ID, 'm', 0.f, States::infected);
    States::infected(INFECTED_STATE_ID, '*', config.inf_to_pat, States::patient);
    States::patient(PATIENT_STATE_ID, '0', config.pat_to_dead, States::dead);
    States::isolated(ISOLATED_STATE_ID, 'i', config.pat_to_dead, States::dead);
    States::dead(DEAD_STATE_ID, ' ', config.dead_to_norm, States::normal);
    States::crit_prob = config.crit_prob;
    States::incubation_time = config.incub_time;

    srand(time(nullptr));

    auto F = Field(config.field_size, config.isol_place);
    F.infect(random() % config.field_size, random() % config.field_size);
//    F.show();

    for (size_t i = 0; i < config.num_of_eras; ++i) {
        if (!((i / 10) % 10)) {
//            F.show();
            auto statistics = F.get_statistics();
            for (auto elem : statistics) {
//                std::cout << " | " << elem.first << ": " << elem.second;
            }
//            std::cout << std::endl;
        }
        F.change_the_era();
    }
    F.show();

    return 0;
}

