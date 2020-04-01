from collections import defaultdict


# from typing import DefaultDict


class State:
    def __init__(self, new_id: int, new_repr: str,
                 prob: defaultdict = None):
        #        prob: DefaultDict[State, float] = None):
        """
        :param new_id: unique id of the state
        :param new_repr: string representation for visualization
        :param prob: probability get other state
        """
        self.id: int = new_id
        self.repr: str = new_repr
        self.prob: defaultdict = prob

    def __hash__(self):
        return hash(id)

    def __getitem__(self, item):
        return self.prob[item]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return self.repr


class States:
    NORMAL = State(0, '.')
    INFECTED = State(1, '*')
    PATIENT = State(2, '0')
    DEAD = State(3, ' ')


def init_states(states_class):
    states_class.NORMAL.prob = defaultdict(float, {States.NORMAL: 1 / 3})
    states_class.INFECTED.prob = defaultdict(float, {States.INFECTED: 1 / 3})
    states_class.PATIENT.prob = defaultdict(float, {States.PATIENT: 1 / 3})
    states_class.DEAD.prob = defaultdict(float, {})


init_states(States)
