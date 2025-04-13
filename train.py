"""
This module contains the implementations of training each model
and saving it to file. A model can be evaluated by running the game
with the model.

Contains:
    ReplayBuffer: A helper class that implements a replay buffer for storing game data
    train: The main training loop for each model
"""

import game
import model

class ReplayBuffer():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    ValModel = model.ValueLearning(6, 2, ReplayBuffer(1000))
    for i in range(500):
        Game = game.game()
        score = Game.run_game(ValModel)
        print("Score: " + str(score))
        ValModel.update()

    ValModel.save("valmodel.pth")