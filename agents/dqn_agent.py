import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Dueling CNN: (2, 6, 7) board -> Q-values for 7 columns.

    Three conv layers capture local spatial patterns (rows, cols, diagonals).
    Dueling heads separate state value from per-action advantages, which
    improves learning stability in 2-player games.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256),
            nn.ReLU(),
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, 7)

    def forward(self, x):
        feat = self.features(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, lr=1e-4, gamma=0.99, device=None):
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber: more robust to large TD errors

    @staticmethod
    def board_to_state(board, player=1):
        """Convert raw board to a (2, 6, 7) float32 array from player's perspective.

        Channel 0: cells occupied by `player`
        Channel 1: cells occupied by the opponent
        """
        state = np.zeros((2, 6, 7), dtype=np.float32)
        state[0] = (board == player).astype(np.float32)
        state[1] = (board == -player).astype(np.float32)
        return state

    def act(self, state, legal_actions, epsilon):
        """Epsilon-greedy action, restricted to legal columns."""
        if np.random.random() < epsilon:
            return int(np.random.choice(legal_actions))

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        # Mask illegal actions with -inf so argmax always picks a legal column
        mask = np.full(7, -np.inf)
        mask[legal_actions] = q_values[legal_actions]
        return int(np.argmax(mask))

    def learn(self, batch, gamma_n: float = None):
        """One gradient step using a sampled batch.

        Args:
            gamma_n: discount for the bootstrap term.  Pass gamma**n when the
                     batch contains n-step returns; defaults to self.gamma (1-step).
        Returns:
            scalar loss value
        """
        states, actions, rewards, next_states, dones = batch
        bootstrap_gamma = gamma_n if gamma_n is not None else self.gamma

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # Current Q(s, a)
        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + bootstrap_gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
