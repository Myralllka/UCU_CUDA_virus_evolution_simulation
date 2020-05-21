//
// Created by botsula on 14.05.20.
//

#ifndef LINEAR_CPP_SYM_FIELD_H
#define LINEAR_CPP_SYM_FIELD_H

#include <iostream>
#include <vector>
#include <functional>
#include "person.h"
#include "state.h"
#include "constants.h"
#include "state_functions.h"
#include "statistics.h"


class Field {
private:
    int incubation_time = 1;
    int field_size;
    std::vector<std::vector<Person>> matrix;
    States states;

public:
    Field(int size, States states_values) {
        field_size = size;
        states = states_values;
        for (int i = 0; i < size; ++i) {
            std::vector<Person> new_vector;
            for (int j = 0; j < size; ++j) {
                new_vector.emplace_back(Person(create_state(states.normal),
                                               create_state(states_values.infected)));
            }
            matrix.emplace_back(new_vector);
        }
    }

    void show() {
        std::string res = "|";
        for (std::vector row : matrix) {
            for (Person p : row) {
                res.append(" " + p.get_repr() + " |");
            }
            res.append("\n|");
        }
        std::cout << res << std::endl;
        std::cout << "========================================" << std::endl;
    }

    Person get_person(int x, int y) {
        return matrix.at(x).at(y);
    }

    void infect(int x, int y) {
        get_person(x, y).set_static_timer(incubation_time);
        get_person(x, y).state = create_state(states.infected);
        get_person(x, y).next_state = create_state(states.infected);

        std::cout << get_person(x, y).state.get_repr();
    }

//        def __iter__(self):
//        for x in range(len(self.matrix)):
//            for y in range(len(self.matrix[x])):
//                yield x, y, self.matrix[x][y]

    std::vector<std::vector<int>> infect_range(int x, int y) {
        std::vector<std::vector<int>> check_range = {{x + 1, y},
                                                     {x,     y + 1},
                                                     {x - 1, y},
                                                     {x,     y - 1}};
        for (int i = 0; i < (int) check_range.size(); ++i) {
            if ((0 <= check_range.at(i).at(0)) && (check_range.at(i).at(0) < (int) matrix.size())
                && (0 <= check_range.at(i).at(1)) &&
                (check_range.at(i).at(1) < (int) matrix[check_range.at(i).at(0)].size())) {
            } else {
                std::cout << "AAAAAAAAA";
                check_range.erase(check_range.begin() + i);
            }
        }

        for (auto a: check_range) {
            std::cout << a[0], a[1];
        }
        return check_range;
    }

    void calculate_interactions() {
        for (int x = 0; x < (int) matrix.size(); ++x) {
            for (int y = 0; y < (int) matrix[x].size(); ++y) {

                auto temp_person = matrix.at(x).at(y);

                if (temp_person.is_alive() && (temp_person.is_infected() || temp_person.is_patient())) {

                    auto may_infected = infect_range(x, y);
                    std::cout << "Matrix size" << matrix.size() << std::endl;

                    for (auto &i : may_infected) {
                        get_person(i[0], i[1]).become_infected(states);
                    };
                }
            }
        }
    }

    void change_the_era() {
        calculate_interactions();
        for (int x = 0; x < (int) matrix.size(); ++x) {
            for (int y = 0; y < (int) matrix[x].size(); ++y) {
                matrix.at(x).at(y).evolute(states);
            }
        }
    }

};


#endif //LINEAR_CPP_SYM_FIELD_H
