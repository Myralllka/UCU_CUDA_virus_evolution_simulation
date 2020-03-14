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

    def get_infected(self):
        self._state = 1

    def become_patient(self):
        self._state = 2

    def die(self):
        self._state = 3

    def recuperate(self):
        self._state = 0

    @property
    def probability_get_patient(self):
        return self._probabilities[self._state] if self._state == 1 else 0

    @property
    def probability_to_die(self):
        return self._probabilities[self._state] if self._state == 2 else 0

    @property
    def probability_get_infected(self):
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

    def infect(self, x, y):
        self.matrix[x][y] = Person(1)

    def change_the_era(self):
        pass

    def make_probability_matrix(self):
        res = [[0 for i in range(len(self.matrix))] for j in range(len(self.matrix[0]))]
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if self.matrix[i][j].is_alive():
                    if self.matrix[i][j].is_infected():
                        pass




if __name__ == "__main__":
    F = Field(10)
    F.infect(2, 2)
    F.show()
    F.change_the_era()
    F.show()
