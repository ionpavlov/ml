from sys import argv
from copy import copy
import math

class Labyrinth:

    ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

    def __init__(self, filename):

        # Read labyrinth description from file
        with open(filename) as f:

            # Read probabilities
            self.p = list(map(float, f.readline().split()))

            # Read map size
            self.height, self.width = map(int, f.readline().split())

            # Read the number of final states and each of them
            final_states_no = int(f.readline().strip())
            self.final_states = {}
            for i in range(final_states_no):
                line = f.readline().strip().split()
                self.final_states[(int(line[0]), int(line[1]))] = float(line[2])

            # Read the reward for a non-terminal state
            self.default_reward = float(f.readline().strip())

            # Read the map (' ' for empty cells, '*' for walls)
            self.states = set()
            for row in range(self.height):
                line = f.readline().strip()
                for col in range(self.width):
                    if line[col] == ' ':
                        self.states.add((row, col))

    def get_all_states(self):
        return copy(self.states)

    def get_valid_actions(self, state):
        return copy(self.ACTIONS) if not self.is_final_state(state) else []

    def is_valid_state(self, state):
        return state in self.states

    def is_final_state(self, state):
        return state in self.final_states

    def get_reward(self, state):
        return self.final_states.get(state, self.default_reward)

    def get_next_states(self, state, action):

        # TODO (1)

        dict = {}
        if self.is_final_state(state):
            return {}

        if action == "RIGHT":
            next_state = (state[0], state[1]+1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[0]
                else:
                    dict[next_state] = self.p[0]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[0]
                else:
                    dict[state] = self.p[0]

            next_state = (state[0]-1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[1]
                else:
                    dict[next_state] = self.p[1]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[1]
                else:
                    dict[state] = self.p[1]

            next_state = (state[0]+1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state] + self.p[2]
                else:
                    dict[next_state] = self.p[2]
            else:
                if state in dict.keys():
                    dict[state] = dict[state] + self.p[2]
                else:
                    dict[state] = self.p[2]

        if action == "LEFT":
            next_state = (state[0], state[1]-1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[0]
                else:
                    dict[next_state] = self.p[0]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[0]
                else:
                    dict[state] = self.p[0]

            next_state = (state[0]-1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[1]
                else:
                    dict[next_state] = self.p[1]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[1]
                else:
                    dict[state] = self.p[1]

            next_state = (state[0]+1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state] + self.p[2]
                else:
                    dict[next_state] = self.p[2]
            else:
                if state in dict.keys():
                    dict[state] = dict[state] + self.p[2]
                else:
                    dict[state] = self.p[2]

        if action == "UP":
            next_state = (state[0]-1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[0]
                else:
                    dict[next_state] = self.p[0]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[0]
                else:
                    dict[state] = self.p[0]

            next_state = (state[0], state[1]+1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[1]
                else:
                    dict[next_state] = self.p[1]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[1]
                else:
                    dict[state] = self.p[1]

            next_state = (state[0], state[1]-1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state] + self.p[2]
                else:
                    dict[next_state] = self.p[2]
            else:
                if state in dict.keys():
                    dict[state] = dict[state] + self.p[2]
                else:
                    dict[state] = self.p[2]

        if action == "DOWN":
            next_state = (state[0]+1, state[1])
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[0]
                else:
                    dict[next_state] = self.p[0]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[0]
                else:
                    dict[state] = self.p[0]

            next_state = (state[0], state[1]+1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state]+self.p[1]
                else:
                    dict[next_state] = self.p[1]
            else:
                if state in dict.keys():
                    dict[state] = dict[state]+self.p[1]
                else:
                    dict[state] = self.p[1]

            next_state = (state[0], state[1]-1)
            if self.is_valid_state(next_state):
                if next_state in dict.keys():
                    dict[next_state] = dict[next_state] + self.p[2]
                else:
                    dict[next_state] = self.p[2]
            else:
                if state in dict.keys():
                    dict[state] = dict[state] + self.p[2]
                else:
                    dict[state] = self.p[2]

        return dict

    def print_utilities(self, U, precision = 3):
        fmt = "%%%d.%df" % (precision + 4, precision)
        wall = "*" * (precision + 4)
        print("Utilities:")
        for r in range(self.height):
            print(" ".join([(fmt % U[(r,c)]) if (r,c) in self.states else wall
                    for c in range(self.width)]))
        print("")

    def print_policy(self, policy):
        fmt = " %%%ds " % max(map(len, self.ACTIONS))
        wall = "*" * max(map(len, self.ACTIONS))
        print("Policy:")
        for r in range(self.height):
            line = [fmt % policy.get((r,c), wall) for c in range(self.width)]
            print(" ".join(line))
        print("")

    def print_rewards(self, precision = 2):
        fmt = "%%%d.%df" % (precision + 4, precision)
        wall = "*" * (precision + 4)
        print("Rewards:")
        for r in range(self.height):
            line = []
            for c in range(self.width):
                if (r,c) in self.states:
                    line.append(
                        fmt % self.final_states.get((r,c), self.default_reward)
                    )
                else:
                    line.append(wall)
            print(" ".join(line))
        print("")

    def distanta(self, U, Uprim):
        S = len(U.keys())

        Sum = 0.0
        for state in self.states:
            Sum = Sum + (U[state]-Uprim[state])*(U[state]-Uprim[state])

        return (float(1)/S)*math.sqrt(Sum)

def value_iteration(game, discount = 0.9, max_diff = 0.0001):

    # TODO (2)

    policy = {state: "UP" for state in game.get_all_states()}
    U = {state: 0 for state in game.get_all_states()}
    Uprim = {state: game.get_reward(state) for state in game.get_all_states()}

    #Utilites
    while game.distanta(U, Uprim) >= max_diff:
        U = copy(Uprim)
        for state in game.states:
            maximSum = None
            for action in game.get_valid_actions(state):
                Sum = 0
                for next_action,T in game.get_next_states(state, action).items():
                    Sum = Sum + U[next_action]*T
                if maximSum is None or maximSum < Sum:
                    maximSum = Sum
                    policy[state] = action
            if maximSum == None:
                maximSum = 0.0
            Uprim[state] = game.get_reward(state) + discount*maximSum

    return U, policy


if __name__ == "__main__":
    l = Labyrinth(argv[1])
    l.print_rewards()
    u, policy = value_iteration(l)
    l.print_utilities(u)
    l.print_policy(policy)
