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
import numpy as np

class ReplayBuffer():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    buffer = ReplayBuffer(10000)  # Larger buffer for more stable learning
    ValModel = model.ValueLearning(6, 2, buffer)
    epsilon = 1.0  # For exploration
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    best_score = 0
    for episode in range(500):
        Game = game.game()
        score = Game.run_game(ValModel)
        
        # Update epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Track best score
        if score > best_score:
            best_score = score
            ValModel.save(f"valmodel_best.pth")
        
        print(f"Episode {episode}, Score: {score}, Best Score: {best_score}, Epsilon: {epsilon:.3f}")
        
        # Save model periodically
        if episode % 50 == 0:
            ValModel.save(f"valmodel_episode_{episode}.pth")
    
    # Save final model
    ValModel.save("valmodel_final.pth")