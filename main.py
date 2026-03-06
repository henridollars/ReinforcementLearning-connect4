from env.connect4_env import Connect4Env

env = Connect4Env()
env.reset()

moves = [0, 1, 0, 1, 0, 1, 0]
player = 1

print("Initial board:")
env.render()

for move in moves:
    state, reward, done, info = env.step(move, player)
    print(f"\nPlayer {player} plays column {move}")
    env.render()

    if done:
        print("\nGame finished!")
        print("Reward:", reward)
        print("Info:", info)
        break

    player *= -1