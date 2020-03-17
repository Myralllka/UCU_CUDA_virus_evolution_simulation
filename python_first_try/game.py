import random


class Person:
    def __init__(self, state=0):
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
        self._characters = ['.', '*', 'O', ' ']
        self._probabilities = [1 / 3, 1 / 4, 1 / 20, 1]

    def is_alive(self):
        return self._state != 3

    def is_patient(self):
        return self._state == 2

    def is_infected(self):
        return self._state == 1

    def is_healthy(self):
        return self._state == 0

    def become_infected(self):
        self._state = 1

    def become_patient(self):
        self._state = 2

    def become_dead(self):
        self._state = 3

    def become_healthy(self):
        self._state = 0

    @property
    def probability_become_patient(self):
        return self._probabilities[self._state] if self._state == 1 else 0

    @property
    def probability_become_dead(self):
        return self._probabilities[self._state] if self._state == 2 else 0

    @property
    def probability_become_infected(self):
        return self._probabilities[self._state] if self._state == 0 else 0

    def __str__(self):
        return self._characters[self._state]


class Field:
    def __init__(self, x: int):
        self.matrix = [[Person() for i in range(x)] for j in range(x)]

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

    def person(self, x, y):
        return self.matrix[x][y]

    def make_probability_matrix(self):
        res = [[0 for i in range(len(self.matrix))] for j in range(len(self.matrix[0]))]
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if self.person(i, j).is_alive():
                    if self.person(i, j).is_infected():
                        for a, b in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                            state = random.choices([0, 1], weights=[self.person(a, b).probability_become_infected,
                                                                    1 - self.person(a, b).probability_become_infected])

                    elif self.person(i, j).is_patient():
                        pass


if __name__ == "__main__":
    F = Field(10)
    F.infect(2, 2, 1)
    F.show()
    F.change_the_era()
    F.show()
