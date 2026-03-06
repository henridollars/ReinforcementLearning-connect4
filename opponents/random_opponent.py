import random


class RandomOpponent:
    def act(self, env):
        legal_actions = env.get_legal_actions()
        return random.choice(legal_actions)
        