from agents import QLearningAgent, RandomAgent
from environment.mario_env import MarioEnvironment
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import config

def main():
    env = MarioEnvironment()
    
    # Choose the agent type
    agent_type = input("Choose agent type (q for Q-Learning, r for Random): ").lower()
    if agent_type == 'q':
        agent = QLearningAgent(action_space=len(env.action_space), state_space=config.STATE_SPACE)
    elif agent_type == 'r':
        agent = RandomAgent(action_space=len(env.action_space))
    else:
        print("Invalid agent type. Defaulting to Q-Learning.")
        agent = QLearningAgent(action_space=len(env.action_space), state_space=config.STATE_SPACE)

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render()
            
            human_action = env.get_human_action()
            if human_action == -1:  # Check for quit signal
                print("Quitting the game.")
                env.close()
                return
            
            if human_action != "NOOP":  # If not NOOP
                action = human_action
                print(f"Human action: {action}")
            else:
                processed_state = agent.preprocess_state(state)
                action = agent.get_action(processed_state)
                print(f"AI action: {action}")
            
            next_state, reward, done, info = env.step(action)
            
            processed_state = agent.preprocess_state(state)
            processed_next_state = agent.preprocess_state(next_state)
            agent.update(processed_state, action, reward, processed_next_state)
            
            state = next_state
            total_reward += reward
            
            print(f"Episode: {episode + 1}, Reward: {reward}")

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()