import copy
import random

from agents.dqn_agent import DQNAgent


class SelfPlayOpponent:
    """Pool of frozen past-agent snapshots used as the self-play opponent.

    Call ``update(agent)`` periodically to add the current policy to the pool.
    ``act(env)`` picks a random snapshot from the pool and returns its action
    for player -1 (the opponent side).
    """

    MAX_POOL = 5

    def __init__(self):
        self._pool: list[DQNAgent] = []

    def update(self, agent: DQNAgent) -> None:
        """Deep-copy the current agent and add it to the pool."""
        clone = copy.deepcopy(agent)
        clone.online_net.eval()
        if len(self._pool) >= self.MAX_POOL:
            self._pool[random.randrange(len(self._pool))] = clone
        else:
            self._pool.append(clone)

    def act(self, env, player: int = -1) -> int:
        if not self._pool:
            return random.choice(env.get_legal_actions())
        opp = random.choice(self._pool)
        # Show the board from the acting player's own perspective
        state = DQNAgent.board_to_state(env.board, player=player)
        legal = env.get_legal_actions()
        # Small epsilon so the pool stays slightly explorative
        return opp.act(state, legal, epsilon=0.05)
