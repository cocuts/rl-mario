# Mario Q-Learning AI

This project implements a Q-Learning AI agent that learns to play Super Mario Bros. The agent can learn from both human gameplay and its own experiences, creating an interactive learning environment.

## Features

- Q-Learning agent that improves its gameplay over time
- Human input support, allowing players to take control at any time
- Seamless transition between AI and human control
- Reward shaping to encourage desirable behaviors in the AI
- Global keyboard input capture for responsive controls

## Requirements

- Python 3.7+
- gym-super-mario-bros
- numpy
- keyboard

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mario-q-learning.git
   cd mario-q-learning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the game:

```
python main.py
```

- Choose the agent type when prompted (Q-Learning or Random).
- Use the following keys to control Mario:
  - 'A': Move left
  - 'D': Move right
  - 'W': Jump
  - 'S': Run/Fire
- Press 'ESC' to quit the game at any time.

The AI will take over if no human input is detected for 5 seconds.

## Project Structure

- `main.py`: The entry point of the program. Handles the game loop and user interface.
- `config.py`: Contains configuration parameters for the game and learning process.
- `agents/qlearning_agent.py`: Implements the Q-Learning agent.
- `agents/random_agent.py`: Implements a random action agent for comparison.
- `environment/mario_env.py`: Wraps the Super Mario Bros environment and handles the interface between the game and the AI/human input.

## Customization

- Adjust learning parameters in `config.py` to experiment with different learning behaviors.
- Modify the reward shaping function in `mario_env.py` to encourage specific behaviors.
- Experiment with different state representations in the `get_state_key` method of `qlearning_agent.py`.

## Contributing

Contributions to improve the AI, add features, or fix bugs are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym for the gym-super-mario-bros environment
- The reinforcement learning community for inspiration and algorithms

Enjoy playing and learning with your Mario AI!
