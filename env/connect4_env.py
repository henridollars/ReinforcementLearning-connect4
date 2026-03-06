#env connect 4
import numpy as np

ROWS = 6
COLS = 7


class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.done = False

    def reset(self):
        """Reset the board to an empty state."""
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.done = False
        return self.board.copy()

    def get_legal_actions(self):
        """Return columns where a move can still be played."""
        return [col for col in range(COLS) if self.board[0, col] == 0]

    def step(self, action, player):
        """
        Play one move for the given player.

        Args:
            action (int): column index from 0 to 6
            player (int): 1 for agent, -1 for opponent

        Returns:
            next_state, reward, done, info
        """
        if self.done:
            raise ValueError("Game is already finished. Call reset() to start a new game.")

        if action not in self.get_legal_actions():
            return self.board.copy(), -10, True, {"illegal_move": True}

        row = self._drop_piece(action, player)

        if self.check_win(player):
            self.done = True
            reward = 1 if player == 1 else -1
            return self.board.copy(), reward, True, {
                "illegal_move": False,
                "winner": player,
                "row": row,
                "col": action,
            }

        if len(self.get_legal_actions()) == 0:
            self.done = True
            return self.board.copy(), 0, True, {
                "illegal_move": False,
                "winner": 0,
                "draw": True,
            }

        return self.board.copy(), 0, False, {
            "illegal_move": False,
            "winner": None,
        }

    def _drop_piece(self, action, player):
        """Drop a piece into the chosen column and return the row index."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                return row
        raise ValueError(f"Column {action} is full.")

    def check_win(self, player):
        """Check whether the given player has 4 in a row."""
        # Horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return True

        # Vertical
        for row in range(ROWS - 3):
            for col in range(COLS):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return True

        # Diagonal down-right
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True

        # Diagonal up-right
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return True

        return False

    def render(self):
        """Print the board in a readable way."""
        symbols = {0: ".", 1: "X", -1: "O"}
        print("\n".join(" ".join(symbols[cell] for cell in row) for row in self.board))
        print("0 1 2 3 4 5 6")