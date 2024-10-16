import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

class LSTMAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = LSTMNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.hidden = self.network.init_hidden(1)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action_values, self.hidden = self.network(state, self.hidden)
        self.network.train()
        return np.argmax(action_values.cpu().data.numpy())

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([s[0] for s in batch]).unsqueeze(1).to(self.device)
        actions = torch.LongTensor([s[1] for s in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([s[3] for s in batch]).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor([s[4] for s in batch]).unsqueeze(1).to(self.device)

        hidden = self.network.init_hidden(self.batch_size)
        q_values, _ = self.network(states, hidden)
        next_q_values, _ = self.network(next_states, hidden)

        q_value = q_values.gather(1, actions)
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.criterion(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def preprocess_state(self, state):
        # Convert the state to a flat array
        return np.array(state).flatten()

    def reset(self):
        self.hidden = self.network.init_hidden(1)