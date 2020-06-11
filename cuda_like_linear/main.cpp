#include <matrix/m_matrix.h>
#include "file_interface/conf_parser.h"
#include <simul_helpers.h>

#define PRINT_DELAY_ITERS 1u
#define NAMED_OUTPUT


ConfigFileOpt parse_conf(int argc, char *argv[]);

float probab_arr[NUMBER_OF_STATES];

int main(int argc, char *argv[]) {
    ConfigFileOpt config = parse_conf(argc, argv);
    srand(time(nullptr)); // set seed form random

    size_t isolation_places = config.isol_place;
    const float tmp_probab_arr[NUMBER_OF_STATES] = {config.healthy_to_infected,         // healthy
                                                    config.infected_to_patient,         // infected
                                                    config.patient_to_crit_patient,     // patient
                                                    config.patient_to_dead,             // patient_crit
                                                    FINAL_NEXT_STATE_PROBAB,            // dead
                                                    FINAL_NEXT_STATE_PROBAB,            //
                                                    FINAL_NEXT_STATE_PROBAB,            //
                                                    FINAL_NEXT_STATE_PROBAB,            //
                                                    FINAL_NEXT_STATE_PROBAB};           // immunity
    for (size_t i = 0u; i < NUMBER_OF_STATES; ++i)
        probab_arr[i] = tmp_probab_arr[i];

    ///////////////////////// INIT WORKING FIELDS //////////////////////////
    // TODO: on CUDA the field array can be already field with 0 == HEALTHY_ID
    std::vector<uint8_t> field_data((2u + config.field_size) * (2u + config.field_size));
    std::vector<uint8_t> next_field_data((2u + config.field_size) * (2u + config.field_size));
    //  vector is initialized to 0 == HEALTHY_ID automatically
//        for (auto &cell : field_data)
//            cell = HEALTHY_ID;

    // fills with IMMUNITY_ID by default
    fill_array_border(field_data.data(), config.field_size + 2u);
    fill_array_border(next_field_data.data(), config.field_size + 2u);

    // actual working fields
    m_matrix<uint8_t> field{config.field_size + 2u, config.field_size + 2u, field_data.data()};
    m_matrix<uint8_t> next_field{config.field_size + 2u, config.field_size + 2u, next_field_data.data()};
    // indicate witch field is current and witch next
    bool next = true;
    // infect first cell
    field.get((random() % config.field_size) + 1u, (random() % config.field_size) + 1u) = INFECTED_ID;
    ///////////////////////// END INIT WORKING FIELDS //////////////////////

    std::cout << PRINT_DELAY_ITERS << std::endl;
    std::cout << config.field_size * config.field_size << std::endl;
    Statistics statistics{};

    /////////////////////////////////////////////////// MAIN LOOP  ////////////////////////////////////////////////////
    for (size_t i = 0u; i < config.num_of_eras / PRINT_DELAY_ITERS; ++i) {
        if (next)
            statistics = get_statistics(field);
        else
            statistics = get_statistics(next_field);
        ///////////////////// OUTPUT OUTLAY ////////////////////
        // normal, immunity, infected, patient, isolated, dead;
#ifdef NAMED_OUTPUT
        std::cout << "healthy " << statistics.healthy << " "
                  << "immunity " << statistics.immunity << " "
                  << "infected " << statistics.infected << " "
                  << "patient " << statistics.patient << " "
                  << "isolated " << statistics.isolated << " "
                  << "dead " << statistics.dead << std::endl;
#else
        std::cout << statistics.healthy << " "
                  << statistics.immunity << " "
                  << statistics.infected << " "
                  << statistics.patient << " "
                  << statistics.isolated << " "
                  << statistics.dead << std::endl;
#endif // NAMED_OUTPUT
        ///////////////////// OUTPUT OUTLAY END ////////////////

        //////////////////////////////////// DELAY LOOP  /////////////////////////////////////
        for (size_t print_delay = 0u; print_delay < PRINT_DELAY_ITERS; ++print_delay) {
            if (statistics.immunity + statistics.dead == config.field_size * config.field_size)
                // finish simulation after system stabilization
                return 0;
            if (next)
                change_the_era(field, next_field, &isolation_places);
            else
                change_the_era(next_field, field, &isolation_places);
            next = !next;
        }
        //////////////////////////////////// DELAY LOOP END //////////////////////////////////
    }
    /////////////////////////////////////////////////// MAIN LOOP END /////////////////////////////////////////////////
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
