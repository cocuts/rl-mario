import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pygame
import config

class MarioEnvironment:
    def __init__(self):
        self.env = gym_super_mario_bros.make(config.ENV_NAME)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.action_space = SIMPLE_MOVEMENT
        
        pygame.init()
        self.clock = pygame.time.Clock()

    def reset(self):
        return self.env.reset()

    def step(self, ai_action):
        human_action = self.get_human_action()
        #print(human_action)
        if human_action == -1:  # Check for quit signal
            return None, None, True, {"quit": True}
        
        if human_action != SIMPLE_MOVEMENT.index(['NOOP']):
            action = human_action
            action_source = "Human"
        else:
            #print(ai_action)
            action = SIMPLE_MOVEMENT.index(['NOOP'])#ai_action #['right', 'A', 'B'] #ai_action
            action_source = "AI"
        
        next_state, reward, done, info = self.env.step(action)
        info["action_source"] = action_source
        info["action_taken"] = SIMPLE_MOVEMENT[action]
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()
        #pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 FPS

    def close(self):
        self.env.close()
        pygame.quit()

    def get_human_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1  # Special value to indicate game should exit

        keys = pygame.key.get_pressed()
        #print(keys)
        if keys[pygame.K_RIGHT] and keys[pygame.K_s] and keys[pygame.K_a]:
            return SIMPLE_MOVEMENT.index(['right', 'A', 'B'])
        elif keys[pygame.K_RIGHT] and keys[pygame.K_s]:
            return SIMPLE_MOVEMENT.index(['right', 'A'])
        elif keys[pygame.K_RIGHT] and keys[pygame.K_a]:
            return SIMPLE_MOVEMENT.index(['right', 'B'])
        elif keys[pygame.K_RIGHT]:
            return SIMPLE_MOVEMENT.index(['right'])
        elif keys[pygame.K_LEFT]:
            return SIMPLE_MOVEMENT.index(['left'])
        elif keys[pygame.K_s]:
            return SIMPLE_MOVEMENT.index(['A'])
        elif keys[pygame.K_a]:
            return SIMPLE_MOVEMENT.index(['B'])
        else:
            return SIMPLE_MOVEMENT.index(['NOOP'])