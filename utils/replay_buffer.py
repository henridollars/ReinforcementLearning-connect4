import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (Schaul et al. 2015).

    Transitions are sampled proportional to |TD error|^alpha.
    Importance-sampling weights correct for the introduced bias.

    Args:
        capacity: maximum number of transitions stored
        alpha:    priority exponent — 0 = uniform, 1 = full prioritization (default 0.6)
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity     = capacity
        self.alpha        = alpha
        self.buffer: list = []
        self.priorities   = np.zeros(capacity, dtype=np.float32)
        self.pos          = 0
        self.size         = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Add a transition; new entries always get max priority."""
        if self.size < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch proportional to priority.

        Returns:
            batch:   (states, actions, rewards, next_states, dones) numpy arrays
            indices: buffer indices of sampled transitions (needed for priority update)
            weights: importance-sampling weights as float32 array, max-normalised
        """
        priorities = self.priorities[: self.size].astype(np.float64)
        probs      = priorities ** self.alpha
        probs     /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        batch   = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # IS weights — correct for sampling bias; anneal beta → 1 over training
        weights  = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        ), indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update stored priorities from latest TD errors."""
        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6   # small epsilon avoids zero priority
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return self.size
