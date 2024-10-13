from pynput import keyboard
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class InputHandler:
    def __init__(self):
        self.key_states = {
            'left': False,
            'right': False,
            'down': False,
            'up': False,
            'a': False,
            'b': False
        }
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'a':
                self.key_states['left'] = True
            elif key.char == 'd':
                self.key_states['right'] = True
            elif key.char == 's':
                self.key_states['down'] = True
            elif key.char == 'w':
                self.key_states['up'] = True
            elif key.char == 'j':
                self.key_states['a'] = True
            elif key.char == 'k':
                self.key_states['b'] = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char == 'a':
                self.key_states['left'] = False
            elif key.char == 'd':
                self.key_states['right'] = False
            elif key.char == 's':
                self.key_states['down'] = False
            elif key.char == 'w':
                self.key_states['up'] = False
            elif key.char == 'j':
                self.key_states['a'] = False
            elif key.char == 'k':
                self.key_states['b'] = False
        except AttributeError:
            pass

    def get_action(self):
        if self.key_states['right']:
            return SIMPLE_MOVEMENT.index([1, 0, 0])  # Right
        elif self.key_states['right'] and self.key_states['a']:
            return SIMPLE_MOVEMENT.index([1, 0, 1])  # Right + A
        elif self.key_states['right'] and self.key_states['b']:
            return SIMPLE_MOVEMENT.index([1, 1, 0])  # Right + B
        elif self.key_states['left']:
            return SIMPLE_MOVEMENT.index([0, 0, 0])  # Left (NOOP in SIMPLE_MOVEMENT)
        elif self.key_states['down']:
            return SIMPLE_MOVEMENT.index([0, 1, 0])  # Down
        elif self.key_states['up']:
            return SIMPLE_MOVEMENT.index([0, 0, 1])  # Up (A button in SIMPLE_MOVEMENT)
        elif self.key_states['a']:
            return SIMPLE_MOVEMENT.index([0, 0, 1])  # A
        elif self.key_states['b']:
            return SIMPLE_MOVEMENT.index([0, 1, 0])  # B
        else:
            return None  # No action (let AI take over)

    def stop(self):
        self.listener.stop()