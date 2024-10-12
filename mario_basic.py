import random
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Set up the Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Define a simple random agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self):
        return random.randint(0, self.action_space.n - 1)

# Create the agent
agent = RandomAgent(env.action_space)

# Run the game
num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.get_action()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Render the game screen
        env.render()
        
        # Add a small delay to make the visualization visible
        time.sleep(0.01)
        
        step += 1
        
        # Print step information
        print(f"Episode: {episode + 1}, Step: {step}, Action: {action}, Reward: {reward}")

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()