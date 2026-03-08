import csv
import os
import random
from collections import deque

from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from env.connect4_env import Connect4Env
from opponents.heuristic_opponent import HeuristicOpponent
from opponents.random_opponent import RandomOpponent
from opponents.self_play_opponent import SelfPlayOpponent
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed

# ── Phase 1: train from scratch vs random ──
PHASE1 = dict(
    n_episodes=30_000,
    opponent_cls=RandomOpponent,
    epsilon_start=1.0,
    epsilon_decay_steps=300_000,
    load_path=None,
    save_path="checkpoints/dqn_phase1.pth",
    self_play=False,
)

# ── Phase 2: noisy heuristic (30 % noise) ──
PHASE2 = dict(
    n_episodes=40_000,
    opponent_cls=lambda: HeuristicOpponent(noise_prob=0.3),
    epsilon_start=0.5,
    epsilon_decay_steps=200_000,
    load_path="checkpoints/dqn_phase1.pth",
    save_path="checkpoints/dqn_phase2.pth",
    self_play=False,
)

# ── Phase 3: full-strength heuristic ──
PHASE3 = dict(
    n_episodes=20_000,
    opponent_cls=lambda: HeuristicOpponent(noise_prob=0.0),
    epsilon_start=0.3,
    epsilon_decay_steps=100_000,
    load_path="checkpoints/dqn_phase2.pth",
    save_path="checkpoints/dqn_phase3.pth",
    self_play=False,
)

# ── Phase 4: self-play ──
PHASE4 = dict(
    n_episodes=50_000,
    opponent_cls=None,          # unused in self-play mode
    epsilon_start=0.2,
    epsilon_decay_steps=300_000,
    load_path="checkpoints/dqn_phase3.pth",
    save_path="checkpoints/dqn_phase4.pth",
    self_play=True,
    # 40 % of episodes use the full-strength heuristic to prevent forgetting;
    # 60 % use the self-play pool to push beyond heuristic-level play.
    heuristic_mix=0.4,
)

# ── Shared hyperparameters, can adapt if needed ──
SEED = 42
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 256
MIN_BUFFER_SIZE = 2_000
LR = 3e-4
GAMMA = 0.99
EPSILON_END = 0.05
TARGET_UPDATE_FREQ = 500
LOG_FREQ = 500
SHAPING_SCALE = 0.1
N_STEP = 5
SELF_PLAY_UPDATE_FREQ = 1_000   # add current agent to pool every N episodes

_ROWS, _COLS = 6, 7


# ── Helper functions ──

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
    """Horizontal mirror: flip board columns and mirror the action index.

    Connect4 is left-right symmetric, so every real transition yields a free
    augmented transition at zero extra game-play cost.
    """
    fs = state[:, :, ::-1].copy()
    fns = next_state[:, :, ::-1].copy()
    fa = (_COLS - 1) - action
    return fs, fa, reward, fns, done


def _curve_path(save_path: str) -> str:
    """Derive a CSV path for the learning curve from the model save path."""
    base = os.path.splitext(os.path.basename(save_path))[0]   # e.g. dqn_phase1
    return os.path.join("checkpoints", f"curve_{base}.csv")


def train(phase: dict):
    """Run one training phase. Pass PHASE1 / PHASE2 / PHASE3 / PHASE4.

    In addition to saving the model checkpoint, writes a learning-curve CSV to
    checkpoints/curve_<phase>.csv with columns:
        episode, win_rate, loss_rate, draw_rate
    sampled every LOG_FREQ episodes (rolling window over the last LOG_FREQ games).
    """
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    n_episodes          = phase["n_episodes"]
    epsilon_start       = phase["epsilon_start"]
    epsilon_decay_steps = phase["epsilon_decay_steps"]
    save_path           = phase["save_path"]
    is_self_play        = phase.get("self_play", False)

    curve_path = _curve_path(save_path)

    env   = Connect4Env()
    agent = DQNAgent(lr=LR, gamma=GAMMA)

    if phase["load_path"]:
        agent.load(phase["load_path"])
        print(f"Loaded weights from {phase['load_path']}")

    buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Build opponent
    heuristic_mix = phase.get("heuristic_mix", 0.0)
    if is_self_play:
        sp_opp           = SelfPlayOpponent()
        sp_opp.update(agent)   # seed pool with initial weights
        heuristic_anchor = HeuristicOpponent(noise_prob=0.0) if heuristic_mix > 0 else None
        opponent         = None
    else:
        sp_opp           = None
        heuristic_anchor = None
        opponent         = phase["opponent_cls"]()

    gamma_n = GAMMA ** N_STEP

    epsilon     = epsilon_start
    total_steps = 0
    outcomes    = []   # 'win' | 'loss' | 'draw' — track real game results

    # Open the learning-curve CSV (overwrite if phase is re-run)
    curve_file = open(curve_path, "w", newline="")
    curve_writer = csv.writer(curve_file)
    curve_writer.writerow(["episode", "win_rate", "loss_rate", "draw_rate"])

    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        # Self-play: periodically snapshot the current policy
        if is_self_play and episode % SELF_PLAY_UPDATE_FREQ == 0:
            sp_opp.update(agent)

        state_raw = env.reset()

        # ── Randomly assign sides so the agent learns both positions ──────────
        # agent_player ∈ {1, -1}; rewards are always converted to agent's frame
        agent_player = 1 if random.random() < 0.5 else -1
        opp_player   = -agent_player

        # If opponent goes first, let it play before the main loop
        if agent_player == -1 and not env.done:
            if is_self_play:
                opp_action = (heuristic_anchor.act(env, opp_player)
                              if heuristic_anchor and random.random() < heuristic_mix
                              else sp_opp.act(env, opp_player))
            else:
                opp_action = opponent.act(env, opp_player)
            state_raw, _, done, _ = env.step(opp_action, opp_player)
            # Extremely rare (opponent wins on move 1?) — just skip episode
            if done:
                outcomes.append("loss")
                continue

        state          = DQNAgent.board_to_state(state_raw, player=agent_player)
        episode_outcome = None
        n_buf: deque   = deque()

        while not env.done:
            # ── Agent's turn ──
            score_before = _score_board(env.board) * agent_player
            legal  = env.get_legal_actions()
            action = agent.act(state, legal, epsilon)
            next_raw, raw_reward, done, info = env.step(action, agent_player)

            # Convert global reward (player-1 frame) to agent's perspective
            agent_reward = raw_reward * agent_player if done else 0.0

            if done:
                winner = info.get("winner")
                episode_outcome = (
                    "win"  if winner == agent_player else
                    "loss" if winner == opp_player   else "draw"
                )
            else:
                # ── Opponent's turn ──
                if is_self_play:
                    opp_action = (heuristic_anchor.act(env, opp_player)
                                  if heuristic_anchor and random.random() < heuristic_mix
                                  else sp_opp.act(env, opp_player))
                else:
                    opp_action = opponent.act(env, opp_player)
                next_raw, raw_reward, done, info = env.step(opp_action, opp_player)

                if done:
                    agent_reward = raw_reward * agent_player
                    winner = info.get("winner")
                    episode_outcome = (
                        "win"  if winner == agent_player else
                        "loss" if winner == opp_player   else "draw"
                    )

            # Intermediate shaping in agent's frame (only for non-terminal steps)
            if not done:
                score_after  = _score_board(next_raw) * agent_player
                agent_reward += SHAPING_SCALE * (score_after - score_before)

            next_state = DQNAgent.board_to_state(next_raw, player=agent_player)
            n_buf.append((state, action, agent_reward, next_state, float(done)))

            # Push to buffer once a full n-step window is ready (+mirrored copy)
            if len(n_buf) == N_STEP:
                trans = _pop_n_step(n_buf, GAMMA)
                buffer.push(*trans)
                buffer.push(*_flip_transition(*trans))

            state        = next_state
            total_steps += 1

            # Learn
            if len(buffer) >= MIN_BUFFER_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.learn(batch, gamma_n=gamma_n)

            # Sync target network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

        # Flush remaining transitions shorter than n steps
        while n_buf:
            trans = _pop_n_step(n_buf, GAMMA)
            buffer.push(*trans)
            buffer.push(*_flip_transition(*trans))

        # Linear step-based epsilon decay
        epsilon = max(
            EPSILON_END,
            epsilon_start - (epsilon_start - EPSILON_END) * total_steps / epsilon_decay_steps,
        )
        outcomes.append(episode_outcome or "draw")

        if episode % LOG_FREQ == 0:
            window = outcomes[-LOG_FREQ:]
            wins   = window.count("win")
            losses = window.count("loss")
            draws  = window.count("draw")
            wr = wins   / LOG_FREQ
            lr = losses / LOG_FREQ
            dr = draws  / LOG_FREQ
            tqdm.write(
                f"Episode {episode:6d} | ε={epsilon:.3f} | "
                f"W/L/D: {wr:.1%}/{lr:.1%}/{dr:.1%} | "
                f"Buffer: {len(buffer)}"
            )
            curve_writer.writerow([episode, f"{wr:.4f}", f"{lr:.4f}", f"{dr:.4f}"])
            curve_file.flush()

    curve_file.close()
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Learning curve saved to {curve_path}")
    return outcomes


if __name__ == "__main__":
    import sys
    phases = {"1": PHASE1, "2": PHASE2, "3": PHASE3, "4": PHASE4}
    phase  = phases.get(sys.argv[1] if len(sys.argv) > 1 else "1", PHASE1)
    train(phase)
