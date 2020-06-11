//
// Created by fenix on 6/10/20.
//

#ifndef LINEAR_CPP_SYM_M_MATRIX_H
#define LINEAR_CPP_SYM_M_MATRIX_H

#include <fstream>
#include <string>

#include <matrix/code_control.h>

#if defined(ERROR_MSG_ON) || defined(DEBUG)

#include <iostream>

#endif

#if defined(ERROR_MSG_ON) || defined(DEBUG)

#include <matrix/index_exception.h>

#endif // DEBUG

template<typename T>
class m_matrix {
    size_t rows{}, cols{};
    T *data = nullptr;

public:
    m_matrix() = delete;

    m_matrix(size_t row, size_t col, T *data) : rows(row), cols(col), data(data) {}

    explicit m_matrix(const std::string &input_filename) {
        cols = 0, rows = 0;
        std::ifstream in(input_filename);
        in >> cols >> rows;
        data = new T[rows * cols * sizeof(T)];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                in >> data[i * cols + j];
            }
        }
    }

    ~m_matrix() = default;

    m_matrix(const m_matrix &matrix) = delete;

    m_matrix(m_matrix &&matrix) noexcept: rows(matrix.rows), cols(matrix.cols), data(matrix.data) {
        matrix.data = nullptr;
    }

    m_matrix &operator=(const m_matrix &matrix) = delete;

    m_matrix &operator=(m_matrix &&matrix) noexcept {
        rows = matrix.rows;
        cols = matrix.cols;
        data = matrix.data;
        matrix.data = nullptr;
        return *this;
    }

    T *get_process_data_portion(size_t process_index, size_t number_of_processes) {
        // return pointer on the beginning of the data for the process
        // the pointer is directly on the data portion, without any overlaps
        auto main_work = rows / number_of_processes;
        auto extra_work = rows % number_of_processes;
        auto process_block_width = (process_index <= extra_work) ? main_work + 1 : main_work;
        return &(data + (process_block_width * cols));
    }

    size_t get_process_data_portion_length(size_t process_index, size_t number_of_processes) {
        auto main_work = rows / number_of_processes;
        auto extra_work = rows % number_of_processes;
        auto process_block_width = (process_index <= extra_work) ? main_work + 1 : main_work;
        return cols * process_block_width;
    }

#if defined(ERROR_MSG_ON) || defined(DEBUG)

    void print_data() {
        for (size_t n = 0; n < cols * rows; ++n) {
            std::cout << data[n] << " ";
        }
    }

#endif // ERROR_MSG_ON || DEBUG

    [[nodiscard]] size_t get_cols() const {
        return cols;
    }

    [[nodiscard]] size_t get_rows() const {
        return rows;
    }

    [[nodiscard]] size_t size() const {
        return rows * cols;
    }

    [[nodiscard]] T &get(size_t row, size_t col) const {
#ifdef DEBUG
        check_indexes(row, col);
#endif // DEBUG
        return data[row * cols + col];
    }

    void set(size_t row, size_t col, const T &element) {
#ifdef DEBUG
        check_indexes(row, col);
#endif // DEBUG
        data[row * cols + col] = element;
    }

    void set(size_t row, size_t col, T &&element) {
#ifdef DEBUG
        check_indexes(row, col);
#endif // DEBUG
        data[row * cols + col] = std::move(element);
    }

    // ITERATOR REALISATION

    const T *begin() const {
        return data;
    }

    const T *end() const {
        return data + cols * rows;
    }

#ifdef DEBUG

    void print() const {
        for (size_t m = 0; m < rows; ++m) {
            for (size_t n = 0; n < cols; ++n) {
                std::cout << data[m * cols + n] << " ";
            }
            std::cout << std::endl;
        }
    }

#endif // DEBUG

private:
#ifdef DEBUG

    void check_indexes(size_t row, size_t col) const {
        if (row > rows or col > cols) {
#ifdef ERROR_MSG_ON
            std::cerr << "Array out of bounds access!" << std::endl;
            std::cerr << (row > rows ? "incorrect number of rows" : "incorrect number of columns") << std::endl;
#endif // ERROR_MSG_ON
            throw IndexException{};
        }
    }

#endif // DEBUG
};

#endif //LINEAR_CPP_SYM_M_MATRIX_H
