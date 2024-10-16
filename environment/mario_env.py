import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from pynput import keyboard
import config

class MarioEnvironment:
    def __init__(self):
        self.env = gym_super_mario_bros.make(config.ENV_NAME)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.action_space = SIMPLE_MOVEMENT
        self.previous_x = 0
        self.previous_y = 0
        self.previous_coins = 0
        self.jump_right_bonus = 0.5  # Bonus for jumping while moving right        
        self.last_human_action_time = time.time()
        self.ai_delay = 5 # 5 seconds delay before AI takes over

        self.current_action = SIMPLE_MOVEMENT.index(['NOOP'])
        self.key_pressed = False
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'd':
                self.current_action = SIMPLE_MOVEMENT.index(['right'])
            elif key.char == 'a':
                self.current_action = SIMPLE_MOVEMENT.index(['left'])
            elif key.char == 'w':
                self.current_action = SIMPLE_MOVEMENT.index(['A'])
            elif key.char == 's':
                self.current_action = SIMPLE_MOVEMENT.index(['B'])
            self.key_pressed = True
            self.last_human_action_time = time.time()
        except AttributeError:
            pass

    def on_release(self, key):
        self.current_action = SIMPLE_MOVEMENT.index(['NOOP'])
        self.key_pressed = False

    def reset(self):
        self.last_human_action_time = time.time()
        return self.env.reset()

    def step(self, ai_action):
        current_time = time.time()
        
        if self.key_pressed:
            action = self.current_action
            action_source = "Human"
        elif current_time - self.last_human_action_time > self.ai_delay:
            action = ai_action
            action_source = "AI"
        else:
            action = SIMPLE_MOVEMENT.index(['NOOP'])
            action_source = "Waiting for human"
        
        next_state, reward, done, info = self.env.step(action)
        shaped_reward = self.shape_reward(info, reward, done, action)

        info["action_source"] = action_source
        info["action_taken"] = SIMPLE_MOVEMENT[action]
        
        return next_state, reward, done, info

    def shape_reward(self, info, reward, done, action):
        
        # Encourage moving right
        x_progress = info['x_pos'] - self.previous_x
        # reward += x_progress
        
        # Encourage jumping while moving right
        y_change = info['y_pos'] - self.previous_y
        if y_change > 0 and x_progress > 0:
            reward += self.jump_right_bonus * min(y_change, x_progress)
        
        # Penalize dying
        if done and info['life'] == 0:
            reward -= 50
        
        # Encourage collecting coins
        reward += (info['coins'] - self.previous_coins) * 10
        
        # Encourage finishing the level
        if info['flag_get']:
            reward += 500
        
        # Additional reward for maintaining speed
        if 'speed' in info:
            reward += info['speed'] * 0.1
        
        self.previous_x = info['x_pos']
        self.previous_y = info['y_pos']
        self.previous_coins = info['coins']
        
    # Add to the __init__ method

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.listener.stop()