import random

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        return random.randint(0, self.action_space - 1)

    def update(self, state, action, reward, next_state):
        # Random agent doesn't learn, so this method does nothing
        pass

    def preprocess_state(self, state):
        # Random agent doesn't need to preprocess the state
        return state