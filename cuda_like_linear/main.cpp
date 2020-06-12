#include "file_interface/conf_parser.h"
#include <simul_helpers.h>

#define PRINT_DELAY_ITERS 1u
//#define NAMED_OUTPUT
//#define DEBUG
//#define TEST_SPEED


#ifdef DEBUG
#include <bitset>
#endif // DEBUG

#ifdef TEST_SPEED
#include <speed_tester.h>
#endif // TEST_SPEED

ConfigFileOpt parse_conf(int argc, char *argv[]);

float probab_arr[NUMBER_OF_STATES];

int main(int argc, char *argv[]) {
#ifdef TEST_SPEED
    auto start = get_current_time_fenced();
#endif // TEST_SPEED

    ConfigFileOpt config = parse_conf(argc, argv);
    srand(static_cast<unsigned int>(time(nullptr))); // set seed form random

    size_t isolation_places = config.isol_place;
    const float tmp_probab_arr[NUMBER_OF_STATES] = {config.healthy_to_infected,         // healthy
                                                    config.infected_to_patient,         // infected
                                                    config.patient_coefficient,         // patient
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

    // actual working fields
    uint8_t *field = field_data.data();
    uint8_t *next_field = next_field_data.data();
    size_t field_side_len = config.field_size + 2u;

    // fills with IMMUNITY_ID by default
    fill_array_border(field, config.field_size + 2u);
    fill_array_border(next_field, config.field_size + 2u);
    // infect first cell
    size_t first_infected_coord = coord((random() % config.field_size) + 1u,
                                        (random() % config.field_size) + 1u,
                                        field_side_len);
    field[first_infected_coord] = INFECTED_ID;
    next_field[first_infected_coord] = INFECTED_ID;

    // indicate witch field is current and witch next
    bool next = true;
    ///////////////////////// END INIT WORKING FIELDS //////////////////////

    std::cout << PRINT_DELAY_ITERS << std::endl;
    std::cout << config.field_size * config.field_size << std::endl;
    Statistics statistics{};

    /////////////////////////////////////////////////// MAIN LOOP  ////////////////////////////////////////////////////
    for (size_t i = 0u; i < config.num_of_eras / PRINT_DELAY_ITERS; ++i) {
        if (next)
            statistics = get_statistics(field, field_side_len);
        else
            statistics = get_statistics(next_field, field_side_len);
        ///////////////////// OUTPUT OUTLAY ////////////////////
        // normal, immunity, infected, patient, isolated, dead;
#ifdef NAMED_OUTPUT
        std::cout << "immunity " << statistics.immunity << " "
                  << "infected " << statistics.infected << " "
                  << "patient " << statistics.patient << " "
                  << "isolated " << statistics.isolated << " "
                  << "dead " << statistics.dead << std::endl;
#else
        std::cout << statistics.immunity << " "
                  << statistics.infected << " "
                  << statistics.patient << " "
                  << statistics.isolated << " "
                  << statistics.dead << std::endl;
#endif // NAMED_OUTPUT

#ifdef DEBUG
        for (size_t row = 1; row < field_side_len - 1; ++row) {
            for (size_t col = 1; col < field_side_len - 1; ++col)
                if (next)
                    std::cout << std::bitset<8>(field[coord(row, col, field_side_len)]) << " ";
                else
                    std::cout << std::bitset<8>(next_field[coord(row, col, field_side_len)]) << " ";
            std::cout << std::endl;
        }
#endif  // DEBUG
        ///////////////////// OUTPUT OUTLAY END ////////////////

        //////////////////////////////////// DELAY LOOP  /////////////////////////////////////
        for (size_t print_delay = 0u; print_delay < PRINT_DELAY_ITERS; ++print_delay) {
            if (statistics.immunity + statistics.dead == config.field_size * config.field_size) {
                // finish simulation after system stabilization
#ifdef TEST_SPEED
                auto finish = get_current_time_fenced();
                std::cout << "Total in s  :\t" << to_s(finish - start) << std::endl;
                std::cout << "Total in ms :\t" << to_ms(finish - start) << std::endl;
#endif // TEST_SPEED
                return 0;
            }
            if (next)
                change_the_era(field, next_field, field_side_len, &isolation_places);
            else
                change_the_era(next_field, field, field_side_len, &isolation_places);
            next = !next;
        }
        //////////////////////////////////// DELAY LOOP END //////////////////////////////////
    }
    /////////////////////////////////////////////////// MAIN LOOP END /////////////////////////////////////////////////
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
