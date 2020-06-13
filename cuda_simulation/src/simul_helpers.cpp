//
// Created by fenix on 6/10/20.
//

#include "simul_helpers.h"

Statistics get_statistics(const uint8_t *field, size_t field_side_len) {
    size_t res[NUMBER_OF_STATES]; // UNUSED_ID index is used for isolated count
    for (auto &re : res)
        re = 0u;

    uint8_t tmp_cell;
    for (size_t row = 1u; row < field_side_len - 1u; ++row)
        for (size_t col = 1u; col < field_side_len - 1u; ++col) {
            tmp_cell = field[coord(row, col, field_side_len)];
//          if (   is_isolated         )
            if (tmp_cell & ISOLATE_MASK)
                ++res[UNUSED_ID];
            else
                ++res[tmp_cell];
        }

    return Statistics{res[HEALTHY_ID],
                      res[IMMUNITY_ID],
                      res[INFECTED_ID] + res[PATIENT_ID] + res[PATIENT_CRIT_ID],
                      res[PATIENT_ID] + res[PATIENT_CRIT_ID] + res[UNUSED_ID],
                      res[UNUSED_ID],
                      res[DEAD_ID]
    };
}

void change_the_era(const uint8_t *field, uint8_t *next_field, size_t field_side_len, size_t *isolation_places) {
    uint8_t cell_st_id;
    for (size_t row = 1u; row < field_side_len - 1u; ++row)
        for (size_t col = 1u; col < field_side_len - 1u; ++col) {
            cell_st_id = field[coord(row, col, field_side_len)];

            if (cell_st_id & FINAL_STATE_CHECK_MASK) {
                next_field[coord(row, col, field_side_len)] = cell_st_id;
                continue;
            } else if (cell_st_id == HEALTHY_ID) {
                infect_cell(field[coord(row - 1u, col, field_side_len)], cell_st_id);
                infect_cell(field[coord(row, col - 1u, field_side_len)], cell_st_id);
                infect_cell(field[coord(row, col + 1u, field_side_len)], cell_st_id);
                infect_cell(field[coord(row + 1u, col, field_side_len)], cell_st_id);
                next_field[coord(row, col, field_side_len)] = cell_st_id;
                continue;
            }
            ///////////////////////////  ISOLATE IF POSSIBLE  ///////////////////////////
            if (cell_st_id & ISOLATE_MASK)
                if (cell_st_id == PATIENT_ID)
                    if (*isolation_places) {
                        --(*isolation_places);
                        cell_st_id = ISOLATE_MASK | PATIENT_ID;
                    }
            ///////////////////////////  ISOLATE IF POSSIBLE END  ///////////////////////

            ///////////////////////////  NEXT STATE  ///////////////////////
            if (random_bool(probab_arr[cell_st_id & REMOVE_ISOL_MASK])) {
                ++cell_st_id;
                if ((cell_st_id & FINAL_STATE_CHECK_MASK) && (cell_st_id & ISOLATE_MASK)) {
                    ++(*isolation_places); // return an isolation place
                    cell_st_id &= REMOVE_ISOL_MASK;
                }
            } else if ((cell_st_id & REMOVE_ISOL_MASK) == PATIENT_CRIT_ID) {
                if (cell_st_id & ISOLATE_MASK)
                    ++(*isolation_places); // return an isolation place
                cell_st_id = IMMUNITY_ID;
            }
            ///////////////////////////  NEXT STATE END  ///////////////////
            next_field[coord(row, col, field_side_len)] = cell_st_id;
        }
}
