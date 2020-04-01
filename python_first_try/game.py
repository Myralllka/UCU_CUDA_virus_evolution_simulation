#!/bin/env python3

from states import States
import random


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

    def is_alive(self) -> bool:
        return self._state != States.DEAD

    def is_patient(self) -> bool:
        return self._state == States.PATIENT

    def is_infected(self):
        return self._state == States.INFECTED

    def is_healthy(self):
        return self._state == States.NORMAL

    def become_infected(self):
        """
        Become ifected only if you are in the normal state
        """
        if self._next_state == States.NORMAL:
            self._next_state = random.choices(
                [self._next_state, States.INFECTED],
                weights=[1 - self._next_state[States.INFECTED],
                         self._next_state[States.INFECTED]])

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
            self._static_state_timer -= 1
        elif self._state == States.PATIENT:
            self._next_state = random.choices(
                [self._next_state, States.INFECTED],
                weights=[1 - self._next_state[States.DEAD],
                         self._next_state[States.DEAD]])

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
        self.person(x, y)._state = States.PATIENT
        self.person(x, y)._next_state = States.PATIENT

    def person(self, x: int, y: int):
        return self.matrix[x][y]

    def __iter__(self):
        for x in range(len(self.matrix)):
            for y in range(len(self.matrix[x])):
                yield x, y, self.matrix[x][y]

    def _infect_range(self, x, y):
        for a, b in filter(lambda a, b: 0 < a < len(self.matrix) and 0 < b < len(self.matrix[a]), 
                ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))):
            yield a, b

    def change_the_era(self):
        for _x, _y, tmp_person in self:
            tmp_person.evolute()

    def calculate_interactions(self):
        for x, y, tmp_person in self:
            if tmp_person.is_alive() and (tmp_person.is_infected() or tmp_person.is_patient()):
                for a, b in self._infect_range(x, y):
                    self.person(a, b).become_infected()
            


if __name__ == "__main__":
    F = Field(10)
    F.infect(2, 2)
    for i in range(3):
        F.show()
        F.change_the_era()
    F.show()