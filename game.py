import pygame
import random

# Size of Game
WINDOW_HEIGHT = 800
WINDOW_WIDTH = int(WINDOW_HEIGHT / 1.5)

# Pipes
NUM_PIPES = 3
PIPE_GAP = 75
PIPE_DISTANCE = int(WINDOW_WIDTH * 0.6)
PIPE_WIDTH = int(WINDOW_WIDTH / 6)
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
GRAVITY = 0.2
bird_velocity = 0
score = 0
scoringTube = 0

def getRandGap():
    toReturn = int((random.random() - 0.5) * 0.75 * WINDOW_HEIGHT)
    print(toReturn)
    return toReturn

# On Creation
pygame.init()
screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
running = True

for i in range(0, NUM_PIPES):
    pipeX.append(WINDOW_WIDTH + i * PIPE_DISTANCE)
    pipeOffset.append(0)
    bottomPipes.append(pygame.rect)
    topPipes.append(pygame.rect)
    pipeOffset.append(getRandGap())

# Game loop
while running:
    # If player quits game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            print("Space bar pressed")
            bird_velocity = -6

    # update pipe x positions
    for i in range(0, NUM_PIPES):
        pipeX[i] = pipeX[i] - pipeVelocity
        if pipeX[i] < -PIPE_WIDTH:
            pipeX[i] = NUM_PIPES * PIPE_DISTANCE
            pipeOffset[i] = getRandGap()

    screen.fill((3, 182, 252))  # draw background

    # draw pipes
    for i in range(0, NUM_PIPES):
        bottomPipes[i] = pygame.Rect(pipeX[i], WINDOW_HEIGHT / 2 + PIPE_GAP - pipeOffset[i], PIPE_WIDTH, WINDOW_HEIGHT / 2 - PIPE_GAP + pipeOffset[i])
        topPipes[i] = pygame.Rect(pipeX[i], 0, PIPE_WIDTH, WINDOW_HEIGHT / 2 - PIPE_GAP - pipeOffset[i])
        pygame.draw.rect(screen, (0, 255, 0), bottomPipes[i])
        pygame.draw.rect(screen, (0, 255, 0), topPipes[i])

    # draw bird
    if bird_y <= WINDOW_HEIGHT or bird_velocity <= 0:
        bird_velocity = bird_velocity + GRAVITY
        bird_y += int(bird_velocity)
    birdCircle = pygame.draw.circle(screen, (255, 0, 0), (BIRD_X, bird_y), BIRD_RADIUS)

    if pipeX[scoringTube] < WINDOW_WIDTH / 2 - PIPE_WIDTH:
        score += 1
        print("score: " + str(score))
        scoringTube += 1
        if scoringTube == NUM_PIPES:
            scoringTube = 0

    pygame.display.flip()
    pygame.time.delay(10)

pygame.quit()
