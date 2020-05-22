//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_FIELD_H
#define LINEAR_CPP_SYM_FIELD_H

#include <vector>
#include "person.h"
#include "../constants.h"
#include <cinttypes>


class Field {
private:
    const uint8_t incubation_time = INCUBATION_TIME;
    std::vector<std::vector<Person>> matrix{};

    struct point {
        size_t x, y;
    };

    [[nodiscard]] std::vector<point> infect_range(size_t x, size_t y) const;

    [[nodiscard]] Person &get_person(const point &p);

    void execute_interactions();

public:
    explicit Field(size_t f_size);

    void show() const;

    void infect(size_t x, size_t y);

//        def __iter__(self):
//        for x in range(len(self.matrix)):
//            for y in range(len(self.matrix[x])):
//                yield x, y, self.matrix[x][y]

    void change_the_era();
};

#endif //LINEAR_CPP_SYM_FIELD_H
