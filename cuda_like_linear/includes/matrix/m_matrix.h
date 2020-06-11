//
// Created by fenix on 6/10/20.
//

#ifndef LINEAR_CPP_SYM_M_MATRIX_H
#define LINEAR_CPP_SYM_M_MATRIX_H

#include <fstream>
#include <string>
#include <code_control.h>

#if defined(ERROR_MSG_ON) || defined(DEBUG)

#include <iostream>
#include <matrix/index_exception.h>

#endif // DEBUG

template<typename T>
class m_matrix {
    size_t rows{}, cols{};
    T *data = nullptr;

public:
    m_matrix() = delete;

    ~m_matrix() = default;

    m_matrix(size_t row, size_t col, T *data);

    explicit m_matrix(const std::string &input_filename);

    m_matrix(const m_matrix<T> &matrix) = delete;

    m_matrix(m_matrix<T> &&matrix) noexcept;

    m_matrix<T> &operator=(const m_matrix<T> &matrix) = delete;

    m_matrix<T> &operator=(m_matrix<T> &&matrix) noexcept;

    [[nodiscard]] T *get_process_data_portion(size_t process_index, size_t number_of_processes);

    [[nodiscard]] size_t get_process_data_portion_length(size_t process_index, size_t number_of_processes);

    [[nodiscard]] size_t get_cols() const;

    [[nodiscard]] size_t get_rows() const;

    [[nodiscard]] size_t size() const;

    [[nodiscard]] T &get(size_t row, size_t col) const;

    void set(size_t row, size_t col, const T &element);

    void set(size_t row, size_t col, T &&element);

    // ITERATOR REALISATION
    const T *begin() const;

    const T *end() const;


#ifdef DEBUG

    void print() const {
        for (size_t m = 0; m < rows; ++m) {
            for (size_t n = 0; n < cols; ++n) {
                std::cout << data[m * cols + n] << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_data() {
        for (size_t n = 0; n < cols * rows; ++n) {
            std::cout << data[n] << " ";
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

template<typename T>
m_matrix<T>::m_matrix(const std::string &input_filename) {
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

template<typename T>
m_matrix<T>::m_matrix(size_t row, size_t col, T *data) : rows(row), cols(col), data(data) {}

template<typename T>
m_matrix<T>::m_matrix(m_matrix &&matrix) noexcept: rows(matrix.rows), cols(matrix.cols), data(matrix.data) {
    matrix.data = nullptr;
}

template<typename T>
m_matrix<T> &m_matrix<T>::operator=(m_matrix &&matrix) noexcept {
    rows = matrix.rows;
    cols = matrix.cols;
    data = matrix.data;
    matrix.data = nullptr;
    return *this;
}

template<typename T>
T *m_matrix<T>::get_process_data_portion(size_t process_index, size_t number_of_processes) {
    // return pointer on the beginning of the data for the process
    // the pointer is directly on the data portion, without any overlaps
    auto main_work = rows / number_of_processes;
    auto extra_work = rows % number_of_processes;
    auto process_block_width = (process_index <= extra_work) ? main_work + 1 : main_work;
    return &(data + (process_block_width * cols));
}

template<typename T>
size_t m_matrix<T>::get_process_data_portion_length(size_t process_index, size_t number_of_processes) {
    auto main_work = rows / number_of_processes;
    auto extra_work = rows % number_of_processes;
    auto process_block_width = (process_index <= extra_work) ? main_work + 1 : main_work;
    return cols * process_block_width;
}

template<typename T>
size_t m_matrix<T>::get_cols() const {
    return cols;
}

template<typename T>
size_t m_matrix<T>::get_rows() const {
    return rows;
}

template<typename T>
size_t m_matrix<T>::size() const {
    return rows * cols;
}

template<typename T>
T &m_matrix<T>::get(size_t row, size_t col) const {
#ifdef DEBUG
    check_indexes(row, col);
#endif // DEBUG
    return data[row * cols + col];
}

template<typename T>
void m_matrix<T>::set(size_t row, size_t col, const T &element) {
#ifdef DEBUG
    check_indexes(row, col);
#endif // DEBUG
    data[row * cols + col] = element;
}

template<typename T>
void m_matrix<T>::set(size_t row, size_t col, T &&element) {
#ifdef DEBUG
    check_indexes(row, col);
#endif // DEBUG
    data[row * cols + col] = std::move(element);
}

template<typename T>
const T *m_matrix<T>::begin() const {
    return data;
}

template<typename T>
const T *m_matrix<T>::end() const {
    return data + cols * rows;
}

#endif //LINEAR_CPP_SYM_M_MATRIX_H
