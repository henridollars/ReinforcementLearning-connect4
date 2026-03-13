import csv
import os
import random
from collections import deque

from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from env.connect4_env import Connect4Env
from opponents.heuristic_opponent import HeuristicOpponent
from opponents.random_opponent import RandomOpponent
from utils.replay_buffer import PrioritizedReplayBuffer
from utils.seed import set_seed

# ── Phase 1: train from scratch vs random ──
PHASE1 = dict(
    n_episodes=25_000,
    make_opponent=lambda: RandomOpponent(),
    epsilon_start=1.0,
    epsilon_end=0.10,
    epsilon_decay_steps=175_000,
    load_path=None,
    save_path="checkpoints/dqn_phase1.pth",
    tactical_shaping=False,
)

# ── Phase 2: noisy heuristic (30% noise) ──
PHASE2 = dict(
    n_episodes=35_000,
    make_opponent=lambda: HeuristicOpponent(noise_prob=0.3),
    epsilon_start=0.40,
    epsilon_end=0.08,
    epsilon_decay_steps=175_000,
    load_path="checkpoints/dqn_phase1.pth",
    save_path="checkpoints/dqn_phase2.pth",
    tactical_shaping=True,
)

# ── Phase 3: mixed opponents ──
# Phase 3 (full-strength heuristic only) was dropped: empirically it caused
# forgetting — win rate vs random fell 15pp and vs heuristic fell 5pp vs Phase 2.
# The mixed distribution below covers full-heuristic exposure (30%) without the
# single-opponent overfitting that wrecked generalisation.
#
# 30% fully random  — prevents forgetting early lessons
# 40% heuristic with noise ~ Uniform(0%, 40%)  — covers mid-skill range
# 30% full heuristic  — maintains peak sharpness
def _make_phase3_opponent():
    r = random.random()
    if r < 0.30:
        return RandomOpponent()
    elif r < 0.70:
        return HeuristicOpponent(noise_prob=random.uniform(0.0, 0.4))
    else:
        return HeuristicOpponent(noise_prob=0.0)

PHASE3 = dict(
    n_episodes=50_000,
    make_opponent=_make_phase3_opponent,
    epsilon_start=0.15,
    epsilon_end=0.02,
    epsilon_decay_steps=225_000,
    load_path="checkpoints/dqn_phase2.pth",
    save_path="checkpoints/dqn_phase3.pth",
    tactical_shaping=True,
)

# ── Shared hyperparameters ──
SEED             = 42
BUFFER_CAPACITY  = 100_000
BATCH_SIZE       = 256
MIN_BUFFER_SIZE  = 2_000
LR               = 3e-4
GAMMA            = 0.99
TARGET_UPDATE_FREQ = 500
LOG_FREQ           = 500
SHAPING_SCALE      = 0.1
WIN_REWARD         = 5.0   # terminal reward for winning
LOSS_REWARD        = 6.0   # terminal penalty for losing (> WIN_REWARD → asymmetric)
DRAW_REWARD        = -0.5  # small penalty to discourage draws
N_STEP             = 5
PER_ALPHA          = 0.6   # priority exponent: 0 = uniform, 1 = full prioritization
PER_BETA_START     = 0.4   # IS weight exponent at start; anneals to 1.0 over training

_ROWS, _COLS = 6, 7


# ── Helper functions ──

def _check_win_at(board, row, col, player):
    """Check whether placing `player` at (row, col) creates 4-in-a-row."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < _ROWS and 0 <= c < _COLS and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False


def _winning_moves(board, player):
    """Return set of columns where `player` would win immediately."""
    wins = set()
    for col in range(_COLS):
        if board[0, col] != 0:
            continue
        for row in range(_ROWS - 1, -1, -1):
            if board[row, col] == 0:
                if _check_win_at(board, row, col, player):
                    wins.add(col)
                break
    return wins


def _score_board(board):
    """Heuristic board score from player 1's perspective.

    +0.5 per open 3-in-a-row (agent)   / -0.5 per open 3-in-a-row (opponent)
    +0.1 per open 2-in-a-row (agent)   / -0.1 per open 2-in-a-row (opponent)
    """
    total = 0.0
    for r in range(_ROWS):
        for c in range(_COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                window = [
                    board[r + i * dr, c + i * dc]
                    for i in range(4)
                    if 0 <= r + i * dr < _ROWS and 0 <= c + i * dc < _COLS
                ]
                if len(window) < 4:
                    continue
                p1 = window.count(1)
                p2 = window.count(-1)
                if p2 == 0:
                    if p1 == 3:
                        total += 0.5
                    elif p1 == 2:
                        total += 0.1
                elif p1 == 0:
                    if p2 == 3:
                        total -= 0.5
                    elif p2 == 2:
                        total -= 0.1
    return total


def _pop_n_step(n_buf: deque, gamma: float):
    """Compute the n-step return for the oldest entry and remove it."""
    R = 0.0
    final_ns, final_done = n_buf[-1][3], n_buf[-1][4]
    for i, (_, _, r, ns, d) in enumerate(n_buf):
        R += (gamma ** i) * r
        if d:
            final_ns, final_done = ns, True
            break
    s0, a0 = n_buf[0][0], n_buf[0][1]
    n_buf.popleft()
    return s0, a0, R, final_ns, final_done


def _flip_transition(state, action, reward, next_state, done):
    """Horizontal mirror: flip board columns and mirror the action index."""
    fs  = state[:, :, ::-1].copy()
    fns = next_state[:, :, ::-1].copy()
    fa  = (_COLS - 1) - action
    return fs, fa, reward, fns, done


def _terminal_reward(raw_reward: float, agent_player: int) -> float:
    """Map environment outcome to asymmetric terminal reward in agent's frame."""
    result = raw_reward * agent_player   # +1 win, -1 loss, 0 draw
    if result > 0:
        return WIN_REWARD
    elif result < 0:
        return -LOSS_REWARD
    else:
        return DRAW_REWARD


def _curve_path(save_path: str) -> str:
    base = os.path.splitext(os.path.basename(save_path))[0]
    return os.path.join("checkpoints", f"curve_{base}.csv")


def train(phase: dict):
    """Run one training phase. Pass PHASE1 / PHASE2 / PHASE3 / PHASE4."""
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    n_episodes          = phase["n_episodes"]
    make_opponent       = phase["make_opponent"]
    epsilon_start       = phase["epsilon_start"]
    epsilon_end         = phase["epsilon_end"]
    epsilon_decay_steps = phase["epsilon_decay_steps"]
    save_path           = phase["save_path"]
    use_tactical        = phase.get("tactical_shaping", False)

    curve_path = _curve_path(save_path)

    env   = Connect4Env()
    agent = DQNAgent(lr=LR, gamma=GAMMA)

    if phase["load_path"]:
        agent.load(phase["load_path"])
        print(f"Loaded weights from {phase['load_path']}")

    buffer  = PrioritizedReplayBuffer(BUFFER_CAPACITY, alpha=PER_ALPHA)
    gamma_n = GAMMA ** N_STEP

    epsilon     = epsilon_start
    total_steps = 0
    outcomes    = []
    ep_rewards  = []   # total shaped reward per episode
    ep_losses   = []   # mean gradient loss per episode (0.0 if no update happened)

    curve_file   = open(curve_path, "w", newline="")
    curve_writer = csv.writer(curve_file)
    curve_writer.writerow(["episode", "win_rate", "loss_rate", "draw_rate",
                           "avg_reward", "avg_loss"])

    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        opponent  = make_opponent()   # fresh opponent sampled each episode
        state_raw = env.reset()

        # Randomly assign sides so the agent learns both positions
        agent_player = 1 if random.random() < 0.5 else -1
        opp_player   = -agent_player

        # If opponent goes first, let it play before the main loop
        if agent_player == -1 and not env.done:
            opp_action = opponent.act(env, opp_player)
            state_raw, _, done, _ = env.step(opp_action, opp_player)
            if done:
                outcomes.append("loss")
                ep_rewards.append(0.0)
                ep_losses.append(0.0)
                continue

        state           = DQNAgent.board_to_state(state_raw, player=agent_player)
        episode_outcome = None
        n_buf: deque    = deque()
        episode_reward  = 0.0
        step_losses     = []

        while not env.done:
            score_before       = _score_board(env.board) * agent_player
            agent_wins_before  = _winning_moves(env.board, agent_player)
            opp_threats_before = _winning_moves(env.board, opp_player)
            legal  = env.get_legal_actions()
            action = agent.act(state, legal, epsilon)
            next_raw, raw_reward, done, info = env.step(action, agent_player)

            agent_reward = _terminal_reward(raw_reward, agent_player) if done else 0.0

            if done:
                winner = info.get("winner")
                episode_outcome = (
                    "win"  if winner == agent_player else
                    "loss" if winner == opp_player   else "draw"
                )
            else:
                opp_action = opponent.act(env, opp_player)
                next_raw, raw_reward, done, info = env.step(opp_action, opp_player)

                if done:
                    agent_reward = _terminal_reward(raw_reward, agent_player)
                    winner = info.get("winner")
                    episode_outcome = (
                        "win"  if winner == agent_player else
                        "loss" if winner == opp_player   else "draw"
                    )

            if not done:
                score_after   = _score_board(next_raw) * agent_player
                agent_reward += SHAPING_SCALE * (score_after - score_before)
                if use_tactical:
                    if agent_wins_before and action not in agent_wins_before:
                        agent_reward -= 1.5   # missed an obvious winning move
                    elif opp_threats_before and action not in opp_threats_before:
                        agent_reward -= 1.0   # failed to block opponent's threat

            episode_reward += agent_reward
            next_state = DQNAgent.board_to_state(next_raw, player=agent_player)
            n_buf.append((state, action, agent_reward, next_state, float(done)))

            if len(n_buf) == N_STEP:
                trans = _pop_n_step(n_buf, GAMMA)
                buffer.push(*trans)
                buffer.push(*_flip_transition(*trans))

            state        = next_state
            total_steps += 1

            if len(buffer) >= MIN_BUFFER_SIZE:
                beta  = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * total_steps / epsilon_decay_steps)
                batch, indices, weights = buffer.sample(BATCH_SIZE, beta=beta)
                loss_val, td_errors = agent.learn(batch, gamma_n=gamma_n, weights=weights)
                buffer.update_priorities(indices, td_errors)
                step_losses.append(loss_val)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

        while n_buf:
            trans = _pop_n_step(n_buf, GAMMA)
            buffer.push(*trans)
            buffer.push(*_flip_transition(*trans))

        # Linear step-based epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps,
        )
        outcomes.append(episode_outcome or "draw")
        ep_rewards.append(episode_reward)
        ep_losses.append(sum(step_losses) / len(step_losses) if step_losses else 0.0)

        if episode % LOG_FREQ == 0:
            window = outcomes[-LOG_FREQ:]
            wins   = window.count("win")
            losses = window.count("loss")
            draws  = window.count("draw")
            wr       = wins   / LOG_FREQ
            lr       = losses / LOG_FREQ
            dr       = draws  / LOG_FREQ
            avg_rew  = sum(ep_rewards[-LOG_FREQ:]) / LOG_FREQ
            avg_loss = sum(ep_losses[-LOG_FREQ:])  / LOG_FREQ
            tqdm.write(
                f"Episode {episode:6d} | ε={epsilon:.3f} | "
                f"W/L/D: {wr:.1%}/{lr:.1%}/{dr:.1%} | "
                f"Avg reward: {avg_rew:.2f} | Avg loss: {avg_loss:.4f}"
            )
            curve_writer.writerow([episode, f"{wr:.4f}", f"{lr:.4f}", f"{dr:.4f}",
                                   f"{avg_rew:.4f}", f"{avg_loss:.6f}"])
            curve_file.flush()

    curve_file.close()
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Learning curve saved to {curve_path}")
    return outcomes


if __name__ == "__main__":
    import sys
    phases = {"1": PHASE1, "2": PHASE2, "3": PHASE3}
    phase  = phases.get(sys.argv[1] if len(sys.argv) > 1 else "1", PHASE1)
    train(phase)
