#include "../../includes/file_interface/conf_parser.h"
#include "../../includes/file_interface/parser_exception.h"
#include <boost/filesystem.hpp>
#include <iostream>

namespace po = boost::program_options;

ConfigFileOpt::ConfigFileOpt() {
    init_opt_description();
}

void ConfigFileOpt::init_opt_description() {
    opt_conf.add_options()
            ("infile", po::value<std::string>(&input_file), "absolute precision")
            ("out_by_a", po::value<std::string>(&out_by_a), "relative precision")
            ("out_by_n", po::value<std::string>(&out_by_n), "parallel execution flows number")
            ("threads", po::value<size_t>(&threads), "initial number of intervals to integrate");
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
