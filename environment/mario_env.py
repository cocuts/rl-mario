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
        info["action_source"] = action_source
        info["action_taken"] = SIMPLE_MOVEMENT[action]
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.listener.stop()