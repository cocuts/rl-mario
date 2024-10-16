from agents import QLearningAgent, RandomAgent
from agents.lstm_agent import LSTMAgent
from environment.mario_env import MarioEnvironment
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import config
import time
from pynput import keyboard

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # Stop listener

def main():
    env = MarioEnvironment()
    
    # Set up a listener for the ESC key to quit the game
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Choose the agent type
    agent_type = input("Choose agent type (q for Q-Learning, r for Random, l for LSTM): ").lower()
    if agent_type == 'q':
        agent = QLearningAgent(action_space=len(env.action_space), state_space=config.STATE_SPACE,
                               learning_rate=0.1, discount_factor=0.99,
                               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995)
    elif agent_type == 'r':
        agent = RandomAgent(action_space=len(env.action_space))
    elif agent_type == 'l':
        state_size = env.env.observation_space.shape[0]  # Adjust this based on your state representation
        agent = LSTMAgent(state_size=state_size, action_size=len(env.action_space))
    else:
        print("Invalid agent type. Defaulting to Q-Learning.")
        agent = QLearningAgent(action_space=len(env.action_space), state_space=config.STATE_SPACE,
                               learning_rate=0.1, discount_factor=0.99,
                               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995)

    print("Get ready! AI will take over in 5 seconds if no action is taken.")
    print("Use 'A' for left, 'D' for right, 'W' for jump, and 'S' for run/fire.")
    print("Press ESC to quit the game.")

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        if isinstance(agent, LSTMAgent):
            agent.reset()  # Reset LSTM hidden state
        done = False
        total_reward = 0
        start_time = time.time()
        
        while not done:
            if not listener.running:
                print("\nQuitting the game.")
                env.close()
                return

            env.render()
            
            current_time = time.time()
            time_left = max(0, 5 - (current_time - start_time))
            if time_left > 0:
                print(f"\rTime until AI takeover: {time_left:.1f} seconds", end="", flush=True)
            
            processed_state = agent.preprocess_state(state)
            ai_action = agent.get_action(processed_state)
            
            next_state, reward, done, info = env.step(ai_action)
            
            if info['action_source'] != "Waiting for human":
                print(f"\nAction taken: {info['action_taken']} by {info['action_source']}")
            
            processed_next_state = agent.preprocess_state(next_state)
            agent.update(processed_state, ai_action, reward, processed_next_state, done)
            
            state = next_state
            total_reward += reward
            
            if info['action_source'] != "Waiting for human":
                print(f"Episode: {episode + 1}, Reward: {reward}")

        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward}")

    env.close()
    listener.stop()

if __name__ == "__main__":
    main()