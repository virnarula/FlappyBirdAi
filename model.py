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
        # Heuristic parameters tuned for current game settings
        self.gravity = 0.2
        self.jump_speed = -6.0
        self.pipe_velocity = 1.0
        self.estimated_pipe_width = 88.0  # ~ WINDOW_WIDTH / 6 when WINDOW_HEIGHT=800
        self.aim_bias = 0.0  # aim near the center of the gap
        self.control_start_distance = 280.0  # start precise control when within this distance
        self.jump_threshold_base = 8.0
        self.cooldown_frames = 10
        self.max_safe_top_distance = 35.0
        self.max_overshoot_margin = 24.0

        self._frames_since_jump = 1000

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        # Emergency floor avoidance
        if dist_to_bottom < 50 and current_velocity > 1.5:
            self._frames_since_jump = 0
            return True

        # Minimal cooldown to prevent constant flapping
        if self._frames_since_jump < self.cooldown_frames:
            self._frames_since_jump += 1
            return False

        # Derived features
        gap_center_relative = (dist_to_opening_top + dist_to_opening_bottom) / 2.0

        # Deadband: if fairly aligned and far from pipe, avoid flapping
        if abs(gap_center_relative - self.aim_bias) < 10.0 and dist_to_pipe > 60:
            self._frames_since_jump += 1
            return False

        # Predict time (in frames) until pipe center reaches bird x
        t_center = (dist_to_pipe + self.estimated_pipe_width / 2.0) / max(0.001, self.pipe_velocity)
        # Clamp horizon for stability
        t_center = max(1.0, min(70.0, t_center))

        # Predicted vertical displacement (positive increases y i.e., going down)
        def displacement(v0: float, t: float) -> float:
            # Use simple discrete-time kinematics approximation
            t_int = int(t)
            return v0 * t_int + 0.5 * self.gravity * t_int * (t_int + 1)

        # Required vertical change to reach gap center
        required_delta = gap_center_relative - self.aim_bias
        # Prediction errors at arrival time
        # Positive error => we end up above center (did not go down enough)
        # Negative error => we end up below center (went down too much)
        delta_nojump = required_delta - displacement(current_velocity, t_center)
        delta_jump = required_delta - displacement(self.jump_speed, t_center)

        # Dynamic threshold: more lenient when close to pipe
        dynamic_threshold = max(6.0, self.jump_threshold_base + 0.02 * t_center)

        should = False

        # Main guidance: if we predict we will be too low at the pipe center, jump
        if dist_to_pipe < self.control_start_distance and dist_to_top > self.max_safe_top_distance:
            # Prefer the action that reduces absolute error by a meaningful margin
            overshoot_penalty = 0.5 * max(0.0, delta_jump)  # penalize ending above center
            improvement = abs(delta_nojump) - (abs(delta_jump) + overshoot_penalty)
            if delta_nojump > dynamic_threshold and improvement > 12.0:
                should = True

        # If descending fast while below target, consider a preemptive jump
        # Remove aggressive preemptive flaps; keep only close-range assist
        if not should and dist_to_pipe < 90 and current_velocity > 4.0 and required_delta < -12.0:
            should = True

        # If very close to the pipe and still far below target, jump
        if not should and 0 <= dist_to_pipe <= self.estimated_pipe_width * 2 and required_delta < -2 * self.jump_threshold_base:
            should = True

        # Avoid ceiling: if we are too close to the top, do not jump
        if should and dist_to_top < self.max_safe_top_distance:
            should = False

        if should:
            self._frames_since_jump = 0
        else:
            self._frames_since_jump += 1

        return should
    
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
        