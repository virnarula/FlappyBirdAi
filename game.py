"""
This module contains the game implementation for Flappy Bird

Contains:
    game: The game class that contains the game logic
        run_game: The main game loop. Can accept a model to play the game
        collision: A function that checks for collisions between the bird and the pipes
        get_rand_gap: A function that generates a random gap for the pipes
        
"""

import pygame
import random
import math
from model import SimpleModel

class game():
    def __init__(self) -> None:
        # Size of Game
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = int(self.WINDOW_HEIGHT / 1.5)

        # Pipes
        self.NUM_PIPES = 3
        self.PIPE_GAP = 75
        self.PIPE_DISTANCE = int(self.WINDOW_WIDTH * 0.6)
        self.PIPE_WIDTH = int(self.WINDOW_WIDTH / 6)
        self.pipeX = []
        self.pipeOffset = []
        self.pipeVelocity = 1
        self.bottomPipes = []
        self.topPipes = []

        # Size of bird
        self.BIRD_RADIUS = 15

        # Movement of bird
        self.BIRD_X = int(self.WINDOW_WIDTH / 2 - self.BIRD_RADIUS / 2)
        self.BIRD_Y = int(self.WINDOW_HEIGHT / 2 - self.BIRD_RADIUS / 2)
        self.GRAVITY = 0.2
        self.bird_velocity = -6
        self.score = 0
        self.scoringTube = 0
        
    def get_rand_gap(self):
        toReturn = int((random.random() - 0.5) * 0.75 * self.WINDOW_HEIGHT)
        return toReturn

    def collision(self, rleft, rtop, width, height,  # rectangle definition
                center_x, center_y, radius):  # circle definition

        # complete boundbox of the rectangle
        rright, rbottom = rleft + width, rtop + height

        # bounding box of the circle
        cleft, ctop = center_x - radius, center_y - radius
        cright, cbottom = center_x + radius, center_y + radius

        # trivial reject if bounding boxes do not intersect
        if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
            return False  # no collision possible

        # check whether any point of rectangle is inside circle's radius
        for x in (rleft, rleft + width):
            for y in (rtop, rtop + height):
                # compare distance between circle's center point and each point of
                # the rectangle with the circle's radius
                if math.hypot(x - center_x, y - center_y) <= radius * 2:
                    return True  # collision detected

        # check if center of circle is inside rectangle
        if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
            return True  # overlaid

        return False  # no collision detected

    def run_game(self, Model):
        # On Creation
        model_given = Model is not None
        pygame.init()
        screen = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])
        running = True
        game_started = model_given  # Start immediately if model is given

        for i in range(0, self.NUM_PIPES):
            self.pipeX.append(self.WINDOW_WIDTH + i * self.PIPE_DISTANCE)
            self.pipeOffset.append(0)
            self.bottomPipes.append(None)
            self.topPipes.append(None)
            self.pipeOffset.append(self.get_rand_gap())

        # Game loop
        while running:
            dist_to_top = 0
            dist_to_bottom = 0
            dist_to_pipe = 0
            dist_to_opening_bottom = 0
            dist_to_opening_top = 0
            
            screen.fill((3, 182, 252))  # draw background

            # draw pipes
            for i in range(0, self.NUM_PIPES):
                self.bottomPipes[i] = pygame.Rect(self.pipeX[i], self.WINDOW_HEIGHT / 2 + self.PIPE_GAP - self.pipeOffset[i], \
                    self.PIPE_WIDTH, self.WINDOW_HEIGHT / 2 - self.PIPE_GAP + self.pipeOffset[i])
                self.topPipes[i] = pygame.Rect(self.pipeX[i], 0, self.PIPE_WIDTH, self.WINDOW_HEIGHT / 2 - self.PIPE_GAP - self.pipeOffset[i])
                pygame.draw.rect(screen, (0, 255, 0), self.bottomPipes[i])
                pygame.draw.rect(screen, (0, 255, 0), self.topPipes[i])
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return self.score
                if not model_given and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game_started = True  # Start the game on space bar press
                        self.bird_velocity = -6

            if game_started:
                # update pipe x positions
                for i in range(0, self.NUM_PIPES):
                    self.pipeX[i] = self.pipeX[i] - self.pipeVelocity
                    
                    # reset pipe if off screen
                    if self.pipeX[i] < -self.PIPE_WIDTH:
                        self.pipeX[i] = self.NUM_PIPES * self.PIPE_DISTANCE
                        self.pipeOffset[i] = self.get_rand_gap()

                # get distances to objects
                dist_to_top = self.BIRD_Y
                dist_to_bottom = self.WINDOW_HEIGHT - self.BIRD_Y
                dist_to_pipe = self.pipeX[self.scoringTube] - self.BIRD_X
                dist_to_opening_bottom = self.bottomPipes[self.scoringTube].topleft[1] - self.BIRD_Y
                dist_to_opening_top = self.topPipes[self.scoringTube].bottomleft[1] - self.BIRD_Y
                
                if model_given and Model.should_jump(dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, self.bird_velocity):
                    self.bird_velocity = -6


                # draw bird
                if self.BIRD_Y <= self.WINDOW_HEIGHT or self.bird_velocity <= 0:
                    self.bird_velocity = self.bird_velocity + self.GRAVITY
                    self.BIRD_Y += int(self.bird_velocity)
            birdCircle = pygame.draw.circle(screen, (255, 0, 0), (self.BIRD_X, self.BIRD_Y), self.BIRD_RADIUS)

            # check if point has been scored
            if self.pipeX[self.scoringTube] < self.WINDOW_WIDTH / 2 - self.PIPE_WIDTH:
                self.score += 1
                print("score: " + str(self.score))
                self.scoringTube += 1
                if self.scoringTube == self.NUM_PIPES:
                    self.scoringTube = 0
                    
            # Check for out of bounds
            if self.BIRD_Y > self.WINDOW_HEIGHT:
                print("Bird hit the floor")
                running = False
            if self.BIRD_Y < 0:
                print("Bird hit the ceiling")
                running = False
            
            # Check for collisions
            for i in range(0, self.NUM_PIPES):
                if self.collision(self.bottomPipes[i].left, self.bottomPipes[i].top, self.bottomPipes[i].width, self.bottomPipes[i].height, self.BIRD_X,
                            self.BIRD_Y, self.BIRD_RADIUS):
                    print("Collision with bottom tube")
                    running = False
                if self.collision(self.topPipes[i].left, self.topPipes[i].top, self.topPipes[i].width, self.topPipes[i].height, self.BIRD_X, self.BIRD_Y,
                            self.BIRD_RADIUS):
                    print("Collision with top tube")
                    running = False

            pygame.display.flip()
            # TODO:: maybe it should not be delaying if a model is given? 
            # or model is given and it is being used for training?
            pygame.time.delay(9)

        if Model is not None:
            pygame.quit()
            return self.score
        else:
            while True:
                # If player quits game
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        pygame.quit()
                        return self.score
        return None

if __name__ == "__main__":
    import argparse
    from model import SimpleModel, ValueLearning, PolicyLearning, Perceptron
    from train import ReplayBuffer
    import torch
    import sys

    parser = argparse.ArgumentParser(description="Run Flappy Bird game with optional model")
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--model_type', type=str, choices=['SimpleModel', 'ValueLearning', 'PolicyLearning', 'Perceptron'], help='Type of the model')
    args = parser.parse_args()

    model = None
    if args.model_type:
        if args.model_type == 'SimpleModel':
            model = SimpleModel()
            if args.model_path:
                print("Warning: SimpleModel does not use a model path. The provided path will be ignored.")
        elif args.model_type == 'ValueLearning':
            if not args.model_path:
                print("Error: ValueLearning requires a model path")
                sys.exit(1)
            model = ValueLearning(state_size=6, action_size=2, replay_buffer=None)
            model.model.load_state_dict(torch.load(args.model_path))
            model.model.eval()
        elif args.model_type == 'PolicyLearning':
            model = PolicyLearning()
            # Implement model loading here
        elif args.model_type == 'Perceptron':
            if not args.model_path:
                print("Error: Perceptron requires a model path")
                sys.exit(1)
            model = Perceptron()
            model.model.load_weights(args.model_path)
    else:
        if args.model_path:
            print("Both --model_path and --model_type must be provided to load a model")
            sys.exit(1)

    Game = game()
    score = Game.run_game(model)
    if score is not None:
        print("Score: " + str(score))
    else:
        print("Game ended. No Score.")