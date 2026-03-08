import random

import numpy as np

from agents.dqn_agent import DQNAgent
from env.connect4_env import Connect4Env
from opponents.heuristic_opponent import HeuristicOpponent
from opponents.random_opponent import RandomOpponent
from utils.seed import set_seed


def _play_game(env, agent, opponent, agent_player: int) -> dict:
    """Play one game.  Returns a result dict with winner, game_length, illegal."""
    state_raw = env.reset()
    opp_player = -agent_player
    moves = 0

    # If agent goes second, let opponent play first
    if agent_player == -1:
        opp_action = opponent.act(env, opp_player)
        state_raw, _, done, info = env.step(opp_action, opp_player)
        moves += 1
        if done:
            winner = info.get("winner")
            return {"winner": winner, "game_length": moves, "illegal": False}

    state = DQNAgent.board_to_state(state_raw, player=agent_player)

    while not env.done:
        legal  = env.get_legal_actions()
        action = agent.act(state, legal, epsilon=0.0)
        next_raw, _, done, info = env.step(action, agent_player)
        moves += 1

        if info.get("illegal_move"):
            return {"winner": opp_player, "game_length": moves, "illegal": True}

        if not done:
            opp_action = opponent.act(env, opp_player)
            next_raw, _, done, info = env.step(opp_action, opp_player)
            moves += 1

        state = DQNAgent.board_to_state(next_raw, player=agent_player)

    winner = info.get("winner")
    return {"winner": winner, "game_length": moves, "illegal": False}


def evaluate(agent, opponent, n_games=200, seed=None, verbose=False):
    """Evaluate the agent against an opponent.

    Plays n_games // 2 games with the agent going first (player 1) and
    n_games // 2 games with the agent going second (player -1).  This ensures
    genuine variety even when both sides are deterministic, and gives a fair
    measure of the agent's ability from both board positions.

    Metrics (per project spec):
      - win / draw / loss rate
      - illegal-move rate  (fraction of agent moves that were illegal)
      - average game length  (moves by both players combined)

    Returns a metrics dict.
    """
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    env = Connect4Env()
    wins = losses = draws = 0
    illegal_moves = 0
    total_agent_moves = 0
    game_lengths = []

    half = n_games // 2
    # First half: agent goes first; second half: agent goes second
    sides = [1] * half + [-1] * (n_games - half)

    for agent_player in sides:
        result = _play_game(env, agent, opponent, agent_player)
        game_lengths.append(result["game_length"])
        # Agent goes first → moves at turns 1,3,5,… → ceil(n/2)
        # Agent goes second → moves at turns 2,4,6,… → floor(n/2)
        if agent_player == 1:
            total_agent_moves += (result["game_length"] + 1) // 2
        else:
            total_agent_moves += result["game_length"] // 2

        if result["illegal"]:
            illegal_moves += 1
            losses += 1
        elif result["winner"] == agent_player:
            wins += 1
        elif result["winner"] == -agent_player:
            losses += 1
        else:
            draws += 1

    illegal_move_rate = illegal_moves / total_agent_moves if total_agent_moves else 0.0
    avg_game_length   = float(np.mean(game_lengths)) if game_lengths else 0.0

    metrics = {
        "win":               wins,
        "loss":              losses,
        "draw":              draws,
        "win_rate":          wins   / n_games,
        "loss_rate":         losses / n_games,
        "draw_rate":         draws  / n_games,
        "illegal_moves":     illegal_moves,
        "total_agent_moves": total_agent_moves,
        "illegal_move_rate": illegal_move_rate,
        "avg_game_length":   avg_game_length,
    }

    if verbose:
        print(f"  Games          : {n_games} "
              f"({half} as first player, {n_games - half} as second)"
              + (f"  seed={seed}" if seed is not None else ""))
        print(f"  Win / Loss / Draw : "
              f"{metrics['win_rate']:.1%} / "
              f"{metrics['loss_rate']:.1%} / "
              f"{metrics['draw_rate']:.1%}  "
              f"({wins} / {losses} / {draws})")
        print(f"  Illegal-move rate : {illegal_move_rate:.4%}  "
              f"({illegal_moves} / {total_agent_moves} agent moves)")
        print(f"  Avg game length   : {avg_game_length:.1f} moves")

    return metrics


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dqn_phase4.pth"
    EVAL_SEED  = 0
    N_GAMES    = 200

    agent = DQNAgent()
    agent.load(checkpoint)

    print(f"\nEvaluating {checkpoint}  ({N_GAMES} games, seed={EVAL_SEED})\n")

    print("vs Random opponent:")
    evaluate(agent, RandomOpponent(), n_games=N_GAMES, seed=EVAL_SEED, verbose=True)

    print("\nvs Heuristic opponent (full strength):")
    evaluate(agent, HeuristicOpponent(noise_prob=0.0), n_games=N_GAMES,
             seed=EVAL_SEED, verbose=True)
