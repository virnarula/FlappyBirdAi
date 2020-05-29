import pygame
import random

# Size of Game
WINDOW_HEIGHT = 800
WINDOW_WIDTH = int(WINDOW_HEIGHT / 1.5)

# Pipes
NUM_PIPES = 3
PIPE_GAP = 100
PIPE_DISTANCE = int(WINDOW_WIDTH * 0.6)
PIPE_WIDTH = int(WINDOW_WIDTH / 8)
pipeX = []
pipeOffset = []
pipeVelocity = 1
bottomPipes = []
topPipes = []

# Size of bird
BIRD_RADIUS = 20

# Movement of bird
BIRD_X = int(WINDOW_WIDTH / 2)
bird_y = int(WINDOW_HEIGHT / 2 - BIRD_RADIUS / 2)
GRAVITY = 0.15
bird_velocity = 0

# On Creation
pygame.init()
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
running = True

for i in range(0, NUM_PIPES):
    pipeX.append(WINDOW_WIDTH + i * PIPE_DISTANCE)
    pipeOffset.append(0)
    bottomPipes.append(pygame.rect)
    topPipes.append(pygame.rect)

# Game loop
while running:
    # If player quits game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update pipe x positions
    for i in range(0, NUM_PIPES):
        pipeX[i] = pipeX[i] - pipeVelocity

    screen.fill((3, 182, 252))  # draw background

    # draw pipes
    for i in range(0, NUM_PIPES):
        bottomPipes[i] = pygame.Rect(pipeX[i], WINDOW_HEIGHT / 2 + PIPE_GAP, PIPE_WIDTH, WINDOW_HEIGHT/2 - PIPE_GAP)
        topPipes[i] = pygame.Rect(pipeX[i], 0, PIPE_WIDTH, WINDOW_HEIGHT/2 - PIPE_GAP)
        pygame.draw.rect(screen, (0, 255, 0), bottomPipes[i])
        pygame.draw.rect(screen, (0, 255, 0), topPipes[i])

    if bird_y <= WINDOW_HEIGHT:
        bird_velocity = bird_velocity + GRAVITY
        bird_y += int(bird_velocity)
    birdCircle = pygame.draw.circle(screen, (255, 0, 0), (BIRD_X, bird_y), BIRD_RADIUS)

    pygame.display.flip()
    pygame.time.delay(10)

pygame.quit()
