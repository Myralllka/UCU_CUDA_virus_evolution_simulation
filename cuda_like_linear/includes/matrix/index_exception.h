//
// Created by fenix on 5/27/20.
//

#ifndef MPI_HEAT_TRANSFER_INDEX_EXCEPTION_H
#define MPI_HEAT_TRANSFER_INDEX_EXCEPTION_H

#include <exception>

class IndexException : public std::exception {
public:
    [[nodiscard]] const char *what() const noexcept override {
        return "Index out of range!";
    }
};

#endif //MPI_HEAT_TRANSFER_INDEX_EXCEPTION_H
