"""
Vision-Guided Dueling Deep Q-Network (DQN) for Personalized Chemotherapy Dosing.
Fuses CT radiomic visual embeddings with clinical state covariates to select optimal
chemotherapy dose fractions across sequential 21-day cycles.
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any

# Clinical Chemotherapy Dose Fractions:
# 0: Drug Holiday (0.0x dose - allow organ recovery)
# 1: Low Metronomic Dose (0.25x)
# 2: Reduced Dose (0.50x)
# 3: Moderate Dose (0.75x)
# 4: Standard Full Dose (1.0x)
# 5: Dose Escalation (1.25x - for aggressive non-responsive tumors with low toxicity)
DOSE_ACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture separating state value V(s) and advantage A(s, a).
    Fuses vision latent vector (z_t) with clinical state vector (s_t).
    """
    def __init__(self, vision_dim: int = 32, clinical_dim: int = 6, num_actions: int = len(DOSE_ACTIONS)):
        super().__init__()
        input_dim = vision_dim + clinical_dim

        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, vision_feat: torch.Tensor, clinical_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuses vision and clinical features.
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        """
        x = torch.cat([vision_feat, clinical_feat], dim=-1)
        feat = self.shared_fc(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        q_vals = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals


class ReplayBuffer:
    """Experience replay buffer for off-policy DQN updates."""
    def __init__(self, capacity: int = 10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        vision_feat: np.ndarray,
        clinical_feat: np.ndarray,
        action: int,
        reward: float,
        next_vision_feat: np.ndarray,
        next_clinical_feat: np.ndarray,
        done: bool
    ):
        self.buffer.append((vision_feat, clinical_feat, action, reward, next_vision_feat, next_clinical_feat, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        v_f, c_f, a, r, next_v_f, next_c_f, d = zip(*transitions)
        return (
            torch.FloatTensor(np.array(v_f)),
            torch.FloatTensor(np.array(c_f)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(next_v_f)),
            torch.FloatTensor(np.array(next_c_f)),
            torch.FloatTensor(d)
        )

    def __len__(self):
        return len(self.buffer)


class ChemoRLAgent:
    """
    Chemotherapy Reinforcement Learning Agent.
    Learns personalized dosing policies from vision + clinical multi-modal state observations.
    """
    def __init__(
        self,
        vision_dim: int = 32,
        clinical_dim: int = 6,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.992,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = len(DOSE_ACTIONS)

        # Main and Target networks
        self.policy_net = DuelingQNetwork(vision_dim, clinical_dim, self.num_actions).to(self.device)
        self.target_net = DuelingQNetwork(vision_dim, clinical_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = ReplayBuffer(capacity=15000)

    def select_action(self, vision_feat: torch.Tensor, clinical_feat: torch.Tensor, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        self.policy_net.eval()
        with torch.no_grad():
            v_t = vision_feat.to(self.device)
            c_t = clinical_feat.to(self.device)
            if v_t.ndim == 1:
                v_t = v_t.unsqueeze(0)
            if c_t.ndim == 1:
                c_t = c_t.unsqueeze(0)
            q_values = self.policy_net(v_t, c_t)
            action = q_values.argmax(dim=-1).item()
        return action

    def get_dose_fraction(self, action_idx: int) -> float:
        return DOSE_ACTIONS[action_idx]

    def update_model(self, batch_size: int = 64) -> float:
        """Executes one step of Double DQN gradient optimization."""
        if len(self.memory) < batch_size:
            return 0.0

        v_f, c_f, a, r, next_v_f, next_c_f, d = self.memory.sample(batch_size)
        v_f = v_f.to(self.device)
        c_f = c_f.to(self.device)
        a = a.unsqueeze(1).to(self.device)
        r = r.unsqueeze(1).to(self.device)
        next_v_f = next_v_f.to(self.device)
        next_c_f = next_c_f.to(self.device)
        d = d.unsqueeze(1).to(self.device)

        # Current Q
        self.policy_net.train()
        q_eval = self.policy_net(v_f, c_f).gather(1, a)

        # Double DQN Target
        with torch.no_grad():
            next_action = self.policy_net(next_v_f, next_c_f).argmax(1, keepdim=True)
            q_next = self.target_net(next_v_f, next_c_f).gather(1, next_action)
            q_target = r + (1.0 - d) * self.gamma * q_next

        loss = nn.SmoothL1Loss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def sync_target(self):
        """Copies weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
