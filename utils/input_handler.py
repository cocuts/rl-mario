import pygame
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class InputHandler:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()

    def get_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1  # Special value to indicate game should exit

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            return SIMPLE_MOVEMENT.index([1, 0, 0])  # Right
        elif keys[pygame.K_RIGHT] and keys[pygame.K_s]:
            return SIMPLE_MOVEMENT.index([1, 0, 1])  # Right + A
        elif keys[pygame.K_RIGHT] and keys[pygame.K_a]:
            return SIMPLE_MOVEMENT.index([1, 1, 0])  # Right + B
        elif keys[pygame.K_LEFT]:
            return SIMPLE_MOVEMENT.index([0, 0, 0])  # Left (NOOP in SIMPLE_MOVEMENT)
        elif keys[pygame.K_DOWN]:
            return SIMPLE_MOVEMENT.index([0, 1, 0])  # Down
        elif keys[pygame.K_UP]:
            return SIMPLE_MOVEMENT.index([0, 0, 1])  # Up (A button in SIMPLE_MOVEMENT)
        elif keys[pygame.K_s]:
            return SIMPLE_MOVEMENT.index([0, 0, 1])  # A
        elif keys[pygame.K_a]:
            return SIMPLE_MOVEMENT.index([0, 1, 0])  # B
        else:
            return None  # No action (let AI take over)

    def stop(self):
        pygame.quit()