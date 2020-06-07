//
// Created by fenix on 5/23/20.
//
#include "objects/field.h"
#include <iostream>
#include "objects/state_obj.h"
#include <unordered_map>


State NoneState = State{};

Field::Field(size_t f_size, size_t isolation_places) : isolation_places(isolation_places) {
    for (size_t i = 0; i < f_size; ++i) {
        std::vector<Person> new_vector(f_size);
        for (size_t j = 0; j < f_size; ++j) {
            new_vector[j] = Person(States::normal, States::normal);
        }
        matrix.emplace_back(new_vector);
    }
    statistics = std::unordered_map<State, size_t, StateHasher>{{States::normal, f_size * f_size}};
    statistics[States::normal] = f_size * f_size;
}

std::vector<Field::point> Field::infect_range(size_t x, size_t y) const {
    const std::vector<point> check_range = {{x + 1, y},
                                            {x,     y + 1},
                                            {x - 1, y},
                                            {x,     y - 1}};
    std::vector<point> res_range{};
    for (auto &check_p : check_range)
        // TODO: check order of boolean operations below
        if (((x == 0 && 1 >= check_p.x) || (x != 0 && check_p.x < matrix.size()))
            && ((y == 0 && 1 >= check_p.y) || (y != 0 && check_p.y < matrix[check_p.x].size())))
            res_range.emplace_back(check_p);
    return res_range;
}

void Field::infect(size_t x, size_t y) {
    get_person(point{x, y}).become_infected();
    ++statistics[States::infected];
    --statistics[States::normal];
}

Person &Field::get_person(const Field::point &p) {
    return matrix[p.x][p.y];
}

void Field::execute_interactions() {
    for (size_t x = 0; x < matrix.size(); ++x)
        for (size_t y = 0; y < matrix[x].size(); ++y) {
            const auto &temp_person = matrix[x][y];
            if (temp_person.is_alive() && ((temp_person.is_infected() || temp_person.is_patient())))
                for (const auto &pos : infect_range(x, y))
                    get_person(pos).try_infect();
        }
}

void Field::change_the_era() {
    execute_interactions();
    for (auto &row : matrix)
        for (auto &person : row) {
            person.evolute(&isolation_places, statistics);
        }
}

std::unordered_map<State, size_t, StateHasher> Field::get_statistics() {
//    std::unordered_map<State, size_t, StateHasher> result{{States::normal, 0}};
//    for (auto &row:matrix) {
//        for (auto &col:row) {
//            ++result[col.get_state()];
//        }
//    }
//    for (auto &index : States::states_vector) {
//        if (*index != States::normal) std::cout << " " << result[*index];
//    }
//    std::cout  << " -----" << std::endl;
    return statistics;
}


void Field::show() const {
    for (const auto &row : matrix) {
        std::cout << "|";
        for (const Person &p : row) {
            std::cout << " " << p.get_repr() << " |";
        }
        std::cout << "\n";
    }
    std::cout << "\n========================================" << std::endl;
}