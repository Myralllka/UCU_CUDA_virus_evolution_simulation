import random
from collections import defaultdict
from typing import DefaultDict


class State:
    def __init__(self, new_id: int, new_repr: str,
                 prob: DefaultDict[str, float] = None):
        """
        :param new_id: unique id of the state
        :param new_repr: string representation for visualization
        :param prob: probability get other state
        """
        self.id: int = new_id
        self.repr: str = new_repr
        self.prob: DefaultDict[str, float] = prob

    def __hash__(self):
        return id

    def __getitem__(self, item):
        return self.prob[item]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return self.repr


def init_states(states_class):
    states_class.NORMAL.prob = defaultdict(float, {States.NORMAL: 1 / 3})
    states_class.INFECTED.prob = defaultdict(float, {States.INFECTED: 1 / 3})
    states_class.PATIENT.prob = defaultdict(float, {States.PATIENT: 1 / 3})
    states_class.DEAD.prob = defaultdict(float, {})


@init_states
class States:
    NORMAL = State(0, '.')
    INFECTED = State(1, '*')
    PATIENT = State(2, '0')
    DEAD = State(3, ' ')


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

    def is_alive(self):
        return self._state != States.DEAD

    def is_patient(self):
        return self._state == States.PATIENT

    def is_infected(self):
        return self._state == States.INFECTED

    def is_healthy(self):
        return self._state == States.NORMAL

    def become_infected(self):
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

    def infect(self, x, y, probability=None):
        if self.person(x, y).is_healthy():
            if probability:
                self.person(x, y).become_infected()

    def change_the_era(self):
        pass

    def person(self, x: int, y: int):
        return self.matrix[x][y]

    def __iter__(self):
        for x in range(len(self.matrix)):
            for y in range(len(self.matrix[x])):
                yield x, y, self.matrix[x][y]

    @staticmethod
    def _infect_range(x, y):
        # TODO: check boundaries
        for a, b in [(x + 1, y), (x, y + 1), (x - 1, y),
                     (x, y - 1)]:
            yield a, b

    def make_probability_matrix(self):
        for x, y, tmp_person in self:
            if not tmp_person.is_alive():
                continue

            if tmp_person.is_infected():
                for a, b in self._infect_range(x, y):
                    self.person(a, b).become_infected()

            elif self.person(x, y).is_patient():
                pass


if __name__ == "__main__":
    F = Field(10)
    F.infect(2, 2, 1)
    F.show()
    F.change_the_era()
    F.show()
