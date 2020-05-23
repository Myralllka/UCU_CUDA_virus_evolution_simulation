//
// Created by fenix on 2/25/20.
//

#ifndef LAB_2_PARALLEL_INTEGRATION_OPTIONSPARSEEXCEPTION_H
#define LAB_2_PARALLEL_INTEGRATION_OPTIONSPARSEEXCEPTION_H


#include <exception>

class OptionsParseException : public std::exception {
public:
    [[nodiscard]] const char *what() const noexcept override {
        return "Invalid configuration file or content!";
    }
};


#endif //LAB_2_PARALLEL_INTEGRATION_OPTIONSPARSEEXCEPTION_H
