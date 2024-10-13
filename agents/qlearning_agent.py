import numpy as np
import config

class QLearningAgent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = config.LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON
        self.q_table = np.zeros((state_space, action_space))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def preprocess_state(self, state):
        flat_state = state.flatten()
        return hash(flat_state.tobytes()) % self.state_space