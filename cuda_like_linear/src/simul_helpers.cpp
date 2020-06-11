//
// Created by fenix on 6/10/20.
//

#include "simul_helpers.h"

Statistics get_statistics(const m_matrix<uint8_t> &field) {
    size_t res[NUMBER_OF_STATES];
    for (auto &re : res)
        re = 0u;

    uint8_t tmp_cell;
    for (size_t row = 1u; row < field.get_rows() - 1u; ++row)
        for (size_t col = 1u; col < field.get_cols() - 1u; ++col) {
            tmp_cell = field.get(row, col);
//          if (   is_isolated         )
            if (tmp_cell & ISOLATE_MASK)
                ++res[UNUSED_ID];
            else
                ++res[tmp_cell];
        }

    return Statistics{res[HEALTHY_ID],
                      res[IMMUNITY_ID],
                      res[INFECTED_ID] + res[PATIENT_ID] + res[PATIENT_CRIT_ID] + res[DEAD_ID],
                      res[PATIENT_ID] + res[PATIENT_CRIT_ID],
                      res[UNUSED_ID],
                      res[DEAD_ID]
    };
}

void change_the_era(const m_matrix<uint8_t> &field, m_matrix<uint8_t> &next_field, size_t *isolation_places) {
    uint8_t cell_st_id;
    for (size_t row = 1u; row < field.get_rows() - 1u; ++row)
        for (size_t col = 1u; col < field.get_cols() - 1u; ++col) {
            cell_st_id = field.get(row, col);

//            if (cell_st_id & FINAL_STATE_CHECK_MASK) {
            if (cell_st_id == DEAD_ID || cell_st_id == IMMUNITY_ID) {
                continue;
            } else if (cell_st_id == HEALTHY_ID) {
                infect_cell(field.get(row - 1u, col), next_field.get(row, col));
                infect_cell(field.get(row, col - 1u), next_field.get(row, col));
                infect_cell(field.get(row, col + 1u), next_field.get(row, col));
                infect_cell(field.get(row + 1u, col), next_field.get(row, col));
                continue;
            }

            ///////////////////////////  ISOLATE IF POSSIBLE  ///////////////////////////
            if (cell_st_id == PATIENT_ID) {
                if (*isolation_places) {
                    --(*isolation_places);
                    cell_st_id = ISOLATE_MASK | PATIENT_ID;
                    next_field.set(row, col, cell_st_id);
                    // TODO: consider to move all set out of if scopes
                }
            }
            ///////////////////////////  ISOLATE IF POSSIBLE END  ///////////////////////
            // TODO: create and use coord(row, col) inline function

            ///////////////////////////  NEXT STATE  ///////////////////////
            if (random_bool(probab_arr[cell_st_id & REMOVE_ISOL_MASK])) {
                ++cell_st_id;
                if (cell_st_id & FINAL_ISOL_STATE_CHECK_MASK) {
                    ++(*isolation_places); // return an isolation place
                    cell_st_id &= REMOVE_ISOL_MASK;
                }
                next_field.set(row, col, cell_st_id);
            } else if ((cell_st_id & REMOVE_ISOL_MASK) == PATIENT_CRIT_ID) {
                if (cell_st_id & ISOLATE_MASK)
                    ++(*isolation_places); // return an isolation place
                next_field.set(row, col, IMMUNITY_ID);
            }
            ///////////////////////////  NEXT STATE END  ///////////////////
        }
}
