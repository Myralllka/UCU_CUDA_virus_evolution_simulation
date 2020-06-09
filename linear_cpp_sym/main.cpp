#include "objects/state_obj.h"
#include "objects/field.h"
#include "file_interface/conf_parser.h"

#define PRINT_DELAY_ITERS 1

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

    States::patient_coef = config.patient_coefficient;
    States::normal(NORMAL_STATE_ID, NORMAL_STATE_CHAR, config.healthy_to_infected, States::infected);
    States::immunity(IM_NORMAL_STATE_ID, IM_NORMAL_STATE_CHAR, .0f, States::immunity);
    States::infected(INFECTED_STATE_ID, INFECTED_STATE_CHAR, config.infected_to_patient, States::patient);
    States::patient(PATIENT_STATE_ID, PATIENT_STATE_CHAR, config.patient_to_dead, States::dead);
    States::isolated(ISOLATED_STATE_ID, ISOLATED_STATE_CHAR, config.patient_to_dead, States::dead);
    States::dead(DEAD_STATE_ID, DEAD_STATE_CHAR, 1, States::dead);

    srand(time(nullptr));

    auto F = Field(config.field_size, config.isol_place);
    F.infect(random() % config.field_size, random() % config.field_size);

    std::cout << PRINT_DELAY_ITERS << std::endl;
    std::cout << config.field_size * config.field_size << std::endl;
    for (size_t i = 0; i < config.num_of_eras; ++i) {
        auto statistics = F.get_statistics();
        // normal, immunity, infected, patient, isolated, dead;
        for (auto &index : States::states_vector) {
            if (*index != States::normal) std::cout << " " << statistics[*index];
        }
        if (statistics[States::immunity] + statistics[States::dead]  == config.field_size * config.field_size)
            break;
        std::cout << std::endl;
        F.change_the_era();
    }

    return 0;
}

