#include "file_interface/conf_parser.h"
#include "file_interface/parser_exception.h"
#include <iostream>

namespace po = boost::program_options;

ConfigFileOpt::ConfigFileOpt() {
    init_opt_description();
}

void ConfigFileOpt::init_opt_description() {
    opt_conf.add_options()
            ("field_size", po::value<size_t>(&field_size), "field size")
            ("num_of_eras", po::value<size_t>(&num_of_eras), "number of eras")
            ("isol_place", po::value<size_t>(&isol_place), "relative precision")
            ("prob.patient_coefficient", po::value<float>(&patient_coefficient),
             "probability to die or get health from ill state")
            ("prob.healthy_to_infected", po::value<float>(&healthy_to_infected), "healthy to infected probability")
//            ("prob.infected_to_patient", po::value<float>(&infected_to_patient), "infected to patient probability")
            ("prob.incubation_period", po::value<size_t>(&incubation_period),
             "period, period to become patient from infected")
            ("prob.patient_to_dead", po::value<float>(&patient_to_dead), "patient to inactive probability");

}

void ConfigFileOpt::parse(const std::string &file_name) {
    try {
        std::ifstream conf_file(assert_file_exist(file_name));
        po::store(po::parse_config_file(conf_file, opt_conf), var_map);
        po::notify(var_map);
    } catch (std::exception &E) {
        std::cerr << E.what() << std::endl;
        throw OptionsParseException();
    }
    if (incubation_period < 2) {
        std::cerr << "Incubation period should be longer than one era!" << std::endl;
        throw OptionsParseException();
    }
    infected_to_patient = 1.f / (static_cast<float>(incubation_period - 1));
}

std::string ConfigFileOpt::assert_file_exist(const std::string &f_name) {
    if (!boost::filesystem::exists(f_name)) {
        throw std::invalid_argument("File " + f_name + " not found!");
    }
    return f_name;
}
