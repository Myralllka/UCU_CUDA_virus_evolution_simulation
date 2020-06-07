#ifndef LINEAR_CPP_SYM_CONFIG_FILE_H
#define LINEAR_CPP_SYM_CONFIG_FILE_H

#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

class ConfigFileOpt {
public:
    ConfigFileOpt();

    ~ConfigFileOpt() = default;

    void parse(const std::string &file_name);

    // declare all parameters
    size_t isol_place = 0;
    size_t field_size = 0;
    size_t num_of_eras = 0;

    float healthy_to_infected = .0f;
    float infected_to_patient = .0f;
    float patient_to_dead = .0f;
    float patient_coefficient = .0f;

private:
    void init_opt_description();

    static std::string assert_file_exist(const std::string &f_name);

    boost::program_options::options_description opt_conf{"Config File Options"};
    boost::program_options::variables_map var_map{};
};

#endif //COUNT_NUMBER_OF_ALL_WORDS_CONFIG_FILE_H
