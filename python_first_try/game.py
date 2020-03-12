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
        self.state = state
        self.characters = ['.', '*', 'O', ' ']
        self.probabilities = [1 / 3, 1 / 4, 1 / 20]

    def get_infected(self):
        self.state = 1

    def become_ill(self):
        self.state = 2

    def die(self):
        self.state = 3

    def recuperate(self):
        self.state = 0

    def __str__(self):
        return self.characters[self.state]

    def change_the_era(self):
        pass


class Cell:
    def __init__(self, person):
        self.person = person

    def __str__(self):
        return str(self.person)


class ProbabilityMatrix:
    def __init__(self):
        pass

    def apply(self, f):
        pass


class Field:
    def __init__(self, x: int):
        self.matrix = [[Cell(Person()) for i in range(x)] for j in range(x)]

    def show(self):
        res = "|"
        for i in self.matrix:
            for j in i:
                res += " " + str(j) + ' |'
            res += '\n|'
        print(res[:-2])

    def change_the_era(self):
        pass


if __name__ == "__main__":
    F = Field(10)
    F.show()
