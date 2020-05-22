#include "includes/objects/state_obj.h"
#include "includes/state_functions.h"
#include "includes/objects/field.h"

int main() {
//  0 - normal state.       .    ->  1/3 INFECTED 1
//  1 - infected state.     *    ->  1   PATIENT  2
//  2 - patient state.      O    ->  1/3 DEAD     3
//  3 - dead state.        ' '   ->  2/3 NORMAL   0

    States::normal(1, '.', 0.3f, States::infected);
    States::infected(2, '*', 1.0f, States::patient);
    States::patient(3, '0', 0.3f, States::dead);
    States::dead(0, ' ', 0.6f, States::normal);

    auto F = Field(5);
    F.infect(2, 2);
    F.show();

    for (int i = 0; i < 5; ++i) {
        F.show();
        F.change_the_era();
    }
    F.show();

    return 0;
}

