//
// Created by fenix on 2/25/20.
//

#include "../../include/option_parser/ConfigFileOpt.h"
#include "../../include/option_parser/OptionsParseException.h"

#include <boost/filesystem.hpp>
#include <cassert>
#include <iostream>

namespace po = boost::program_options;

ConfigFileOpt::ConfigFileOpt() {
    init_opt_description();
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
    assert_valid_opt_vals();
}

void ConfigFileOpt::init_opt_description() {
    opt_conf.add_options()
            ("abs_pres", po::value<double>(&abs_pars), "absolute precision")
            ("rel_pres", po::value<double>(&rel_pres), "relative precision")
            ("flow_num", po::value<uint16_t>(&flow_num), "parallel execution flows number")
            ("init_steps", po::value<size_t>(&init_steps), "initial number of intervals to integrate")
            ("max_steps", po::value<size_t>(&max_steps), "max number of intervals to integrate")
            ("x.start", po::value<double>(&(x.first)), "start of the x region")
            ("x.end", po::value<double>(&(x.second)), "end of the x region")
            ("y.start", po::value<double>(&(y.first)), "start of the y region")
            ("y.end", po::value<double>(&(y.second)), "end of the y region")
            ("m", po::value<size_t>(&m), "parameter m")
            ("a1", po::value<std::vector<double>>(&a1)->multitoken()->composing(), "parameter a1")
            ("a2", po::value<std::vector<double>>(&a2)->multitoken()->composing(), "parameter a2")
            ("c", po::value<std::vector<double>>(&c)->multitoken()->composing(), "parameter c");
}

std::string ConfigFileOpt::assert_file_exist(const std::string &f_name) {
    if (!boost::filesystem::exists(f_name)) {
        throw std::invalid_argument("File " + f_name + " not found!");
    }
    return f_name;
}

void ConfigFileOpt::assert_valid_opt_vals() const {
    assert(x.first < x.second && "Invalid integration range for x");
    assert(y.first < y.second && "Invalid integration range for y");
}
