//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_FIELD_H
#define LINEAR_CPP_SYM_FIELD_H

#include <vector>
#include "person.h"
#include <cinttypes>
#include <map>

class Field {
    size_t isolation_places;
    std::vector<std::vector<Person>> matrix{};

    struct point {
        size_t x, y;
    };

    [[nodiscard]] std::vector<point> infect_range(size_t x, size_t y) const;

    [[nodiscard]] Person &get_person(const point &p);

    void execute_interactions();

public:
    explicit Field(size_t f_size, size_t isolation_places);

    void show() const;

    void infect(size_t x, size_t y);

    void change_the_era();

    std::map<State, size_t> get_statistics();
};

#endif //LINEAR_CPP_SYM_FIELD_H
