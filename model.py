"""
This module contains various implementations of models for Flappy Bird

Contains implementations of:
    Model: Base model class that defines the interface for game models
    SimpleModel: A simple rule-based model
    ValueLearning: A model that learns using value-based reinforcement learning
    PolicyLearning: A model that learns using policy-based reinforcement learning
    Perceptron: A simple perceptron model
"""

import torch
import numpy as np
 
# Abstract class for interfacing with game
class Model():
    def __init__(self) -> None:
        pass

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        pass
    
class SimpleModel(Model):
    def __init__(self) -> None:
        pass

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        if dist_to_bottom < 100 and current_velocity > 0:
            return True
        
        if dist_to_opening_bottom > 100 and current_velocity > 30:
            return True
    
class ValueLearning(Model):
    def __init__(self, state_size, action_size, replay_buffer) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = replay_buffer
        self.epsilon = 1.0  # Start with 100% exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return bool(np.random.randint(2))  # Random action
            
        with torch.no_grad():
            inputs = np.array([dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity])
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.reshape(1, 6)
            q_values = self.model(inputs)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return q_values[0, 1] > q_values[0, 0]  
        
    def update(self, batch_size=128):
        assert self.replay_buffer is not None, "Replay buffer is not initialized"
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * max_next_q
        
        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

class PolicyLearning(Model):
    def __init__(self) -> None:
        pass

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        pass

class Perceptron(Model):
    def __init__(self) -> None:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        self.model = Sequential([
            Dense(1, input_dim=6, activation='sigmoid')
        ])

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        import numpy as np
        inputs = np.array([dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity])
        inputs = inputs.reshape(1, 6)
        prediction = self.model(inputs)
        return prediction > 0.5
    
    def train(self, data, label):
        pass
        