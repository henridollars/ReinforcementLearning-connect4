import random

from env.connect4_env import Connect4Env
from opponents.random_opponent import RandomOpponent


env = Connect4Env()
opponent = RandomOpponent()

state = env.reset()
player = 1

print("Initial board:")
env.render()

while not env.done:
    if player == 1:
        action = random.choice(env.get_legal_actions())
    else:
        action = opponent.act(env)

    state, reward, done, info = env.step(action, player)

    print(f"\nPlayer {player} plays column {action}")
    env.render()

    if done:
        print("\nGame finished!")
        print("Reward:", reward)
        print("Info:", info)
        break

    player *= -1