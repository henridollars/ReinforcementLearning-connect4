"""serve_agent.py — Flask backend for the Connect 4 DQN agent.

Run:
    pip install flask flask-cors
    python serve_agent.py                          # loads dqn_phase4.pth by default
    python serve_agent.py checkpoints/dqn_phase3.pth   # custom checkpoint

Endpoints:
    POST /move      { "board": [[...6x7...]], "legal": [0,1,...], "player": 1 }
                 -> { "col": 3, "q_values": [0.1, 0.4, ...] }

    GET  /health -> { "status": "ok", "checkpoint": "..." }
"""

import sys
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Path setup: allow running from project root ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "checkpoints/dqn_phase3.pth"
checkpoint = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
PORT = int(os.environ.get("PORT", 5000))

# ── Load agent ────────────────────────────────────────────────────────────────
print(f"Loading agent from: {checkpoint}")
agent = DQNAgent()
agent.load(checkpoint)
agent.online_net.eval()
print("Agent ready.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from the visualizer (any origin)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "checkpoint": checkpoint})


@app.route("/move", methods=["POST"])
def get_move():
    """Return the agent's greedy move and all Q-values for the position.

    Request JSON:
        board  : 6×7 nested list of ints (0 = empty, 1 = agent, -1 = human)
        legal  : list of legal column indices
        player : which player the agent is acting as (1 or -1, default 1)

    Response JSON:
        col      : chosen column (int)
        q_values : list of 7 floats — raw Q-values (masked to -inf for illegal)
    """
    data = request.get_json(force=True)

    board  = np.array(data["board"], dtype=int)         # (6, 7)
    legal  = list(data["legal"])                        # e.g. [0,1,2,4,5,6]
    player = int(data.get("player", 1))

    state  = DQNAgent.board_to_state(board, player=player)
    col    = agent.act(state, legal, epsilon=0.0)

    # Also return raw Q-values so the frontend can show score bars
    import torch
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q = agent.online_net(state_t).squeeze(0).cpu().numpy()

    # Mask illegal actions for the frontend
    masked = q.tolist()
    for c in range(7):
        if c not in legal:
            masked[c] = None   # frontend renders these as "—"

    return jsonify({"col": int(col), "q_values": masked})


if __name__ == "__main__":
    print(f"Starting server on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
