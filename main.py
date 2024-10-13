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
            
            processed_state = agent.preprocess_state(state)
            ai_action = agent.get_action(processed_state)
            
            next_state, reward, done, info = env.step(ai_action)
            
            if info.get("quit", False):
                print("Quitting the game.")
                env.close()
                return
            
            print(f"Action taken: {info['action_taken']} by {info['action_source']}")
            
            processed_next_state = agent.preprocess_state(next_state)
            agent.update(processed_state, ai_action, reward, processed_next_state)
            
            state = next_state
            total_reward += reward
            
            print(f"Episode: {episode + 1}, Reward: {reward}")

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()