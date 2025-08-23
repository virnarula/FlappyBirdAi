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
import torch.nn as nn
from collections import deque
import random
 
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
    def __init__(self,
                 state_size: int = 6,
                 action_size: int = 2,
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 buffer_capacity: int = 100000,
                 batch_size: int = 256,
                 min_buffer_size: int = 2000,
                 target_update_interval: int = 1000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 50000,
                 device: str = None) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_interval = target_update_interval

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Experience replay
        self.replay = deque(maxlen=buffer_capacity)

        # Epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.total_steps = 0

        # Episode tracking
        self._last_state = None
        self._last_action = None
        self._is_training = True

        # Heuristic scales for normalization
        self._norm_scales = np.array([800.0, 800.0, 600.0, 800.0, 800.0, 10.0], dtype=np.float32)
        # Jump cooldown management to stabilize early training
        self._jump_cooldown_frames = 8
        self._frames_since_jump = 999

    def set_training(self, is_training: bool) -> None:
        self._is_training = is_training
        self.q_network.train(is_training)
        self.target_network.train(False)

    def _epsilon(self) -> float:
        # Exponential-style decay over steps
        progress = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - progress)

    def _to_state(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity):
        state = np.array([
            dist_to_top,
            dist_to_bottom,
            dist_to_pipe,
            dist_to_opening_bottom,
            dist_to_opening_top,
            current_velocity,
        ], dtype=np.float32)
        # Normalize to roughly [-1, 1]
        state = state / self._norm_scales
        return state

    def _select_action(self, state_np: np.ndarray) -> int:
        self.total_steps += 1
        if self._is_training and random.random() < self._epsilon():
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def should_jump(self, dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity) -> bool:
        state = self._to_state(dist_to_top, dist_to_bottom, dist_to_pipe, dist_to_opening_bottom, dist_to_opening_top, current_velocity)
        action = self._select_action(state)
        # Enforce minimal cooldown between jumps
        if action == 1 and self._frames_since_jump < self._jump_cooldown_frames:
            action = 0
        # Track cooldown
        if action == 1:
            self._frames_since_jump = 0
        else:
            self._frames_since_jump += 1
        self._last_state = state
        self._last_action = action
        return action == 1  # 1 => jump, 0 => no jump

    def observe(self, reward: float, done: bool, next_state_tuple) -> None:
        if self._last_state is None or self._last_action is None:
            # First frame of episode; nothing to store yet
            return
        next_state = self._to_state(*next_state_tuple)
        self.replay.append((self._last_state, self._last_action, reward, next_state, float(done)))
        self._last_state = next_state
        if done:
            self._last_state = None
            self._last_action = None
        # Learn
        if self._is_training and len(self.replay) >= self.min_buffer_size:
            self._learn_from_replay()

    def _learn_from_replay(self) -> None:
        batch_size = min(self.batch_size, len(self.replay))
        batch = random.sample(self.replay, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_network(states_t).gather(1, actions_t)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodic target update
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def begin_episode(self) -> None:
        self._last_state = None
        self._last_action = None

    def save(self, filename: str) -> None:
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename: str) -> None:
        state = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(state)
        self.target_network.load_state_dict(self.q_network.state_dict())

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
        