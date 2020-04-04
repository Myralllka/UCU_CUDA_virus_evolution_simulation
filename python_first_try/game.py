#!/bin/env python3

import random

from constants import INCUBATION_TIME
from states import States, Statistics


class Person:
    def __init__(self, state=States.NORMAL):
        """
        state:
            0 - normal state.       .
            1 - infected state.     *
            2 - patient state.      O
            3 - dead state.        ' '
        probabilities to change the state:
            - probability get infected
            - probability get ill
            - probability to die
        """
        self._state = state
        self._next_state = state
        self._static_state_timer = 0

    def get_state(self):
        return self._state

    def is_alive(self) -> bool:
        return self._state != States.DEAD

    def is_patient(self) -> bool:
        return self._state == States.PATIENT

    def is_infected(self):
        return self._state == States.INFECTED

    def is_healthy(self):
        return self._state == States.NORMAL

    def become_infected(self) -> bool:
        """
        Become infected only if you are in the normal state
        """
        if self._next_state == States.NORMAL:
            self._next_state = random.choices(
                [self._next_state, States.INFECTED],
                weights=[1 - self._next_state[States.INFECTED],
                         self._next_state[States.INFECTED]])[0]
            if States.INFECTED == self._next_state:
                self._static_state_timer = INCUBATION_TIME
                return True
        return False

    def become_patient(self):
        self._next_state = States.PATIENT

    def become_dead(self):
        self._next_state = States.DEAD

    def become_healthy(self):
        self._next_state = States.NORMAL

    @property
    def probability_become_patient(self):
        return self._state[States.PATIENT]

    @property
    def probability_become_dead(self):
        return self._state[States.DEAD]

    @property
    def probability_become_infected(self):
        return self._state[States.INFECTED]

    def evolute(self):
        if self._state == States.INFECTED:
            if self._static_state_timer > 0:
                self._static_state_timer -= 1
            else:
                self._next_state = States.PATIENT
        elif self._state == States.PATIENT:
            self._next_state = random.choices(
                [States.NORMAL, States.DEAD],
                weights=[1 - self._next_state[States.DEAD],
                         self._next_state[States.DEAD]])[0]

        self._state = self._next_state

    def __str__(self):
        return str(self._state)


class Field:
    def __init__(self, size: int):
        self.matrix = [[Person() for _x in range(size)] for _y in range(size)]

    def show(self):
        res = "|"
        for i in self.matrix:
            for j in i:
                res += " " + str(j) + ' |'
            res += '\n|'
        print(res[:-2])
        print("_" * len(self.matrix) * 4)

    def infect(self, x, y):
        self.person(x,
                    y)._static_state_timer = INCUBATION_TIME
        self.person(x, y)._state = States.INFECTED
        self.person(x, y)._next_state = States.INFECTED

    def person(self, x: int, y: int):
        return self.matrix[x][y]

    def __iter__(self):
        for x in range(len(self.matrix)):
            for y in range(len(self.matrix[x])):
                yield x, y, self.matrix[x][y]

    def _infect_range(self, x, y):
        for a, b in filter(
                lambda tuple2: (0 <= tuple2[0] < len(self.matrix) and 0
                                <= tuple2[1] < len(self.matrix[tuple2[0]])),
                ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))):
            yield a, b

    def change_the_era(self):
        self.calculate_interactions()
        for _x, _y, tmp_person in self:
            tmp_person.evolute()

    def calculate_interactions(self):
        for x, y, tmp_person in self:
            if tmp_person.is_alive() and (
                    tmp_person.is_infected() or tmp_person.is_patient()):
                for a, b in self._infect_range(x, y):
                    self.person(a, b).become_infected()

    def get_statistics(self) -> Statistics:
        stats = Statistics()
        for _x, _y, person in self:
            if person.is_healthy():
                stats.normal += 1
            elif person.is_infected():
                stats.infected += 1
            elif person.is_patient():
                stats.patient += 1
            elif not person.is_alive():
                stats.dead += 1
            else:
                raise ValueError("Error: Not register state!")
        return stats


if __name__ == "__main__":
    F = Field(10)
    F.infect(1, 1)
    for i in range(20):
        F.show()
        F.change_the_era()
    F.show()
