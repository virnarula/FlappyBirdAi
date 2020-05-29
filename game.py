import pygame
import random

# Size of Game
WINDOW_HEIGHT = 800
WINDOW_WIDTH = int(WINDOW_HEIGHT / 1.5)

# Size of Pipes
OBSTACLE = WINDOW_WIDTH

pygame.init()

screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
