#ifndef LINEAR_CPP_SYM_CONFIG_FILE_H
#define LINEAR_CPP_SYM_CONFIG_FILE_H

#include <string>
#include <boost/program_options.hpp>

class ConfigFileOpt {
public:
    ConfigFileOpt();

    ~ConfigFileOpt() = default;

    void parse(const std::string &file_name);

    // make getters

    std::string get_infile() const { return input_file; }

    std::string get_out_by_a() const { return out_by_a; }

    std::string get_out_by_n() const { return out_by_n; }

    size_t get_number_of_threads() const { return threads; }

private:
    void init_opt_description();

    static std::string assert_file_exist(const std::string &f_name);

    // declare all parameters
    std::string input_file;
    std::string out_by_a;
    std::string out_by_n;
    size_t threads = 0;

    boost::program_options::options_description opt_conf{"Config File Options"};
    boost::program_options::variables_map var_map{};
};

#endif //COUNT_NUMBER_OF_ALL_WORDS_CONFIG_FILE_H
