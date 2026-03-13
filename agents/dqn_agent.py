import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Dueling CNN with oriented win-detection kernels: (2, 6, 7) -> Q-values for 7 columns.

    Two parallel branches are concatenated before the linear head:
      - 3×3 branch: captures local spatial context (same as before)
      - oriented branch: 1×4 / 4×1 / 4×4 kernels with direct inductive bias
        toward horizontal, vertical and diagonal 4-in-a-row patterns

    On a 6×7 board the oriented branch output sizes are:
        horizontal (1×4): 16 × 6 × 4  = 384
        vertical   (4×1): 16 × 3 × 7  = 336
        diagonal   (4×4): 16 × 3 × 4  = 192   (covers both diagonal directions)
    """

    # Pre-computed flat sizes for the oriented branch
    _H_FLAT = 16 * 6 * 4   # 384
    _V_FLAT = 16 * 3 * 7   # 336
    _D_FLAT = 16 * 3 * 4   # 192
    _ORIENTED_FLAT = _H_FLAT + _V_FLAT + _D_FLAT   # 912

    def __init__(self):
        super().__init__()

        # ── 3×3 branch (local context) ──
        self.conv_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )   # → 64 × 6 × 7 = 2688

        # ── Oriented win-detection branch ──
        self.horizontal = nn.Conv2d(2, 16, kernel_size=(1, 4))
        self.vertical   = nn.Conv2d(2, 16, kernel_size=(4, 1))
        self.diagonal   = nn.Conv2d(2, 16, kernel_size=(4, 4))

        # ── Shared head ──
        self.head = nn.Sequential(
            nn.Linear(2688 + self._ORIENTED_FLAT, 256),
            nn.ReLU(),
        )
        self.value     = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, 7)

    def forward(self, x):
        conv_feat = self.conv_branch(x)
        h_feat = torch.flatten(torch.relu(self.horizontal(x)), 1)
        v_feat = torch.flatten(torch.relu(self.vertical(x)),   1)
        d_feat = torch.flatten(torch.relu(self.diagonal(x)),   1)
        feat = self.head(torch.cat([conv_feat, h_feat, v_feat, d_feat], dim=1))
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

    def learn(self, batch, gamma_n: float = None, weights=None):
        """One gradient step using a sampled batch.

        Args:
            gamma_n: discount for the bootstrap term.  Pass gamma**n when the
                     batch contains n-step returns; defaults to self.gamma (1-step).
            weights: importance-sampling weights from PrioritizedReplayBuffer
                     (float32 array of shape [batch_size]).  None = uniform.
        Returns:
            (loss_value, td_errors): scalar loss and per-transition |TD error|
                                     array for priority updates.
        """
        states, actions, rewards, next_states, dones = batch
        bootstrap_gamma = gamma_n if gamma_n is not None else self.gamma

        states_t      = torch.tensor(states).to(self.device)
        actions_t     = torch.tensor(actions).to(self.device)
        rewards_t     = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t       = torch.tensor(dones).to(self.device)

        # Current Q(s, a)
        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q       = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards_t + bootstrap_gamma * next_q * (1.0 - dones_t)

        # Per-transition Huber loss — weighted by IS weights if PER is active
        errors = F.smooth_l1_loss(q_values, targets, reduction="none")
        if weights is not None:
            weights_t = torch.tensor(weights).to(self.device)
            loss = (weights_t * errors).mean()
        else:
            loss = errors.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        td_errors = (q_values - targets).detach().abs().cpu().numpy()
        return loss.item(), td_errors

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
