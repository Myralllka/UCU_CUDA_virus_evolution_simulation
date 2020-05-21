#include <iostream>
#include <vector>
#include "includes/state.h"
#include "includes/state_functions.h"
#include "includes/field.h"

int main() {
    States states;
    set_struct(states);

    auto F = Field(5, states);
    F.infect(2,2);
    F.show();

    for (int i = 0; i < 5; ++i) {
        F.show();
        F.change_the_era();
    }
    F.show();

    return 0;
}
