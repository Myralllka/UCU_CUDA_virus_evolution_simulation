//
// Created by fenix on 6/10/20.
//

#include "simul_helpers.h"

Statistics get_statistics(const m_matrix<uint8_t> &field) {
    size_t res[IMMUNITY_ID + 1];
    for (auto &re : res)
        re = 0;

    for (size_t row = 1; row < field.get_rows() - 1; ++row)
        for (size_t col = 1; col < field.get_cols() - 1; ++col)
            ++res[field.get(row, col)];

    return Statistics{res[HEALTHY_ID],
                      res[IMMUNITY_ID],
                      res[INFECTED_ID] + res[PATIENT_ID] + res[PATIENT_CRIT_ID] + res[DEAD_ID],
                      res[PATIENT_ID] + res[PATIENT_CRIT_ID],
                      0,
                      res[DEAD_ID]
    };
}

void change_the_era(const m_matrix<uint8_t> &field, m_matrix<uint8_t> &next_field) {
    uint8_t tmp_cell;
    for (size_t row = 1; row < field.get_rows() - 1; ++row)
        for (size_t col = 1; col < field.get_cols() - 1; ++col) {
            // infect neighbours if able
            if (INFECTED_CHECK_MASK & field.get(row, col)) {
                infect_cell(field.get(row - 1, col), next_field.get(row - 1, col));
                infect_cell(field.get(row, col - 1), next_field.get(row, col - 1));
                infect_cell(field.get(row, col + 1), next_field.get(row, col + 1));
                infect_cell(field.get(row + 1, col), next_field.get(row + 1, col));
            }
            tmp_cell = field.get(row, col);
            // update state (except healthy)
            if (tmp_cell) {
                if (random_bool(probab_arr[tmp_cell]))
                    next_field.get(row, col) = tmp_cell + 1;
                else if (tmp_cell == PATIENT_CRIT_ID)
                    next_field.get(row, col) = IMMUNITY_ID;
            }
        }
}
