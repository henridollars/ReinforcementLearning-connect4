import random


class RandomOpponent:
    def act(self, env, player: int = -1):
        legal_actions = env.get_legal_actions()
        return random.choice(legal_actions)
        