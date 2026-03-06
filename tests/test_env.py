from env.connect4_env import Connect4Env
import numpy as np


def test_reset():
    env = Connect4Env()
    board = env.reset()
    assert np.all(board == 0)


def test_legal_actions_initial():
    env = Connect4Env()
    env.reset()
    assert env.get_legal_actions() == [0, 1, 2, 3, 4, 5, 6]


def test_drop_piece():
    env = Connect4Env()
    env.reset()
    env.step(0, 1)
    assert env.board[5, 0] == 1


def test_full_column_becomes_illegal():
    env = Connect4Env()
    env.reset()

    players = [1, -1, 1, -1, 1, -1]
    for p in players:
        env.step(0, p)

    assert 0 not in env.get_legal_actions()


def test_horizontal_win():
    env = Connect4Env()
    env.reset()
    env.board[5, 0:4] = 1
    assert env.check_win(1)


def test_vertical_win():
    env = Connect4Env()
    env.reset()
    env.board[2:6, 0] = 1
    assert env.check_win(1)


def test_diagonal_win_down_right():
    env = Connect4Env()
    env.reset()
    env.board[5, 0] = 1
    env.board[4, 1] = 1
    env.board[3, 2] = 1
    env.board[2, 3] = 1
    assert env.check_win(1)


def test_diagonal_win_up_right():
    env = Connect4Env()
    env.reset()
    env.board[2, 0] = 1
    env.board[3, 1] = 1
    env.board[4, 2] = 1
    env.board[5, 3] = 1
    assert env.check_win(1)


def test_illegal_move():
    env = Connect4Env()
    env.reset()

    players = [1, -1, 1, -1, 1, -1]
    for p in players:
        env.step(0, p)

    state, reward, done, info = env.step(0, 1)

    assert done is True
    assert reward == -10
    assert info["illegal_move"] is True