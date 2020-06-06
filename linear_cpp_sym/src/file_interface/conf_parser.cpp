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
            // CAN BE CHANGED TO `crit_prob`
            ("prob.patient_coefficient", po::value<float>(&crit_prob), "probability to die or get health from ill state")
            ("prob.healthy_to_infected", po::value<float>(&norm_to_inf), "healthy to infected probability")
            ("prob.infected_to_patient", po::value<float>(&inf_to_pat), "infected to patient probability")
            ("prob.patient_to_dead", po::value<float>(&pat_to_dead), "patient to inactive probability")
            ("prob.dead_to_norm", po::value<float>(&dead_to_norm), "inactive to healthy probability");

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
}

std::string ConfigFileOpt::assert_file_exist(const std::string &f_name) {
    if (!boost::filesystem::exists(f_name)) {
        throw std::invalid_argument("File " + f_name + " not found!");
    }
    return f_name;
}
