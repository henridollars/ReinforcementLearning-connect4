import random

ROWS = 6
COLS = 7

# Columns ordered from center outward, preferred when no tactical play
_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


class HeuristicOpponent:
    """Rule-based opponent.  Priority order:

      1. Win immediately if possible.
      2. Block the opponent's immediate win.
      3. Create a fork (move that sets up two winning threats simultaneously).
      4. Block opponent fork.
      5. Create any 3-in-a-row threat (center-preferred).
      6. Prefer center columns.

    Args:
        noise_prob: probability of ignoring the heuristic and playing randomly.
                    Use 0.3 for Phase 2 warm-up, 0.0 for full strength.
    """

    def __init__(self, noise_prob: float = 0.0):
        self.noise_prob = noise_prob

    def act(self, env, player=-1):
        legal = env.get_legal_actions()

        if self.noise_prob > 0.0 and random.random() < self.noise_prob:
            return random.choice(legal)

        # 1. Win if possible
        for col in legal:
            if self._would_win(env.board, col, player):
                return col

        # 2. Block opponent's immediate win
        opponent = -player
        for col in legal:
            if self._would_win(env.board, col, opponent):
                return col

        # 3. Create a fork: a move that sets up two winning threats at once
        for col in legal:
            if self._count_threats(env.board, col, player) >= 2:
                return col

        # 4. Block opponent fork
        for col in legal:
            if self._count_threats(env.board, col, opponent) >= 2:
                return col

        # 5. Create any 3-in-a-row threat
        for col in _CENTER_ORDER:
            if col in legal and self._count_threats(env.board, col, player) >= 1:
                return col

        # 6. Prefer center columns
        for col in _CENTER_ORDER:
            if col in legal:
                return col

        return random.choice(legal)

    @staticmethod
    def _would_win(board, col, player):
        """Return True if dropping `player`'s piece into `col` wins the game."""
        # Find the row where the piece would land
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                temp = board.copy()
                temp[row, col] = player
                # Inline win check to avoid creating a full env object
                return HeuristicOpponent._check_win(temp, player)
        return False  # column is full

    @staticmethod
    def _check_win(board, player):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r, c + i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(board[r + i, c] == player for i in range(4)):
                    return True
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True
        # Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    @staticmethod
    def _count_threats(board, col, player) -> int:
        """Count how many winning threats `player` would have after playing `col`.

        A threat is a window of 4 with exactly 3 of `player`'s pieces and 1 empty.
        """
        row = -1
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                row = r
                break
        if row == -1:
            return 0  # column full

        temp = board.copy()
        temp[row, col] = player
        count = 0
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    window = []
                    for i in range(4):
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            window.append(temp[nr, nc])
                    if len(window) == 4 and window.count(player) == 3 and window.count(0) == 1:
                        count += 1
        return count
