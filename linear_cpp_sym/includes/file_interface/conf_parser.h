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
    size_t incub_time = 0;
    size_t isol_place = 0, field_size = 0, num_of_eras = 0;
//    float crit_prob = .0f;
    float norm_to_inf = .0f, inf_to_pat = .0f, pat_to_dead = .0f, dead_to_norm = .0f;
    //    size_t threads = 0;

private:
    void init_opt_description();

    static std::string assert_file_exist(const std::string &f_name);

    boost::program_options::options_description opt_conf{"Config File Options"};
    boost::program_options::variables_map var_map{};
};

#endif //COUNT_NUMBER_OF_ALL_WORDS_CONFIG_FILE_H
