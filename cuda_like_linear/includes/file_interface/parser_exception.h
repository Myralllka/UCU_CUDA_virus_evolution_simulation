//
// Created by myralllka on 3/25/20.
//

#ifndef LINEAR_CPP_SYM_PARSER_EXCEPTION_H
#define LINEAR_CPP_SYM_PARSER_EXCEPTION_H

#include <exception>

class OptionsParseException : public std::exception {
public:
    [[nodiscard]] const char *what() const noexcept override {
        return "Invalid configuration file or content!";
    }
};


#endif //COUNT_NUMBER_OF_ALL_WORDS_PARSER_EXCEPTION_H
