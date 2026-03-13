import { useState, useEffect, useCallback, useRef } from "react";

const API_URL = "http://localhost:5000";

const ROWS = 6;
const COLS = 7;
const EMPTY = 0;

function createBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));
}

function dropPiece(board, col, player) {
  const b = board.map(r => [...r]);
  for (let row = ROWS - 1; row >= 0; row--) {
    if (b[row][col] === EMPTY) { b[row][col] = player; return { board: b, row }; }
  }
  return { board: b, row: -1 };
}

function checkWin(board, player) {
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS - 3; c++)
      if ([0,1,2,3].every(i => board[r][c+i] === player)) return true;
  for (let r = 0; r < ROWS - 3; r++)
    for (let c = 0; c < COLS; c++)
      if ([0,1,2,3].every(i => board[r+i][c] === player)) return true;
  for (let r = 0; r < ROWS - 3; r++)
    for (let c = 0; c < COLS - 3; c++)
      if ([0,1,2,3].every(i => board[r+i][c+i] === player)) return true;
  for (let r = 3; r < ROWS; r++)
    for (let c = 0; c < COLS - 3; c++)
      if ([0,1,2,3].every(i => board[r-i][c+i] === player)) return true;
  return false;
}

function getLegal(board) {
  return Array.from({ length: COLS }, (_, c) => c).filter(c => board[0][c] === EMPTY);
}

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  bg:        "#080b14",
  board:     "#0b1428",
  rim:       "#162040",
  hole:      "#050a18",
  holeRim:   "#1a2d5a",
  yellow:    "#fbbf24",
  yellowGlow:"#f59e0b",
  red:       "#f43f5e",
  redGlow:   "#e11d48",
  best:      "#22d3ee",
  text:      "#e2e8f0",
  muted:     "#475569",
  good:      "#4ade80",
};

// P1 (+1) = yellow, P2 (-1) = red — fixed, independent of human/agent assignment
function pieceColor(cellValue) {
  if (cellValue ===  1) return { fill: C.yellow, glow: C.yellowGlow };
  if (cellValue === -1) return { fill: C.red,    glow: C.redGlow    };
  return { fill: C.hole, glow: null };
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function fetchAgentMove(board, legal, player) {
  const res = await fetch(`${API_URL}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board, legal, player }),
  });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json();
}

async function checkServerHealth() {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch { return false; }
}

function qToColor(q, minQ, maxQ) {
  if (q === null) return C.muted;
  const norm = maxQ === minQ ? 0.5 : (q - minQ) / (maxQ - minQ);
  return `hsl(${norm * 120}, 80%, 52%)`;
}

// ── Side-selection screen ─────────────────────────────────────────────────────
function SideSelect({ onSelect }) {
  return (
    <div style={{
      minHeight: "100vh", background: C.bg,
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 32,
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
    }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 10, letterSpacing: 8, color: C.best, marginBottom: 6 }}>DQN AGENT · LIVE</div>
        <div style={{ fontSize: 32, fontWeight: 800, color: C.text, letterSpacing: 4 }}>CONNECT FOUR</div>
      </div>

      <div style={{ fontSize: 12, color: C.muted, letterSpacing: 3, textTransform: "uppercase" }}>
        Choose your side
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        <button onClick={() => onSelect(1)} style={{
          background: "transparent", border: `2px solid ${C.yellow}`,
          color: C.yellow, padding: "18px 32px", cursor: "pointer",
          fontSize: 11, letterSpacing: 3, textTransform: "uppercase",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
          boxShadow: `0 0 20px ${C.yellow}33`,
        }}>
          <div style={{ width: 44, height: 44, borderRadius: "50%", background: C.yellow, boxShadow: `0 0 16px ${C.yellowGlow}` }} />
          Yellow
          <span style={{ fontSize: 9, opacity: 0.7 }}>GOES FIRST</span>
        </button>

        <button onClick={() => onSelect(-1)} style={{
          background: "transparent", border: `2px solid ${C.red}`,
          color: C.red, padding: "18px 32px", cursor: "pointer",
          fontSize: 11, letterSpacing: 3, textTransform: "uppercase",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
          boxShadow: `0 0 20px ${C.red}33`,
        }}>
          <div style={{ width: 44, height: 44, borderRadius: "50%", background: C.red, boxShadow: `0 0 16px ${C.redGlow}` }} />
          Red
          <span style={{ fontSize: 9, opacity: 0.7 }}>GOES SECOND</span>
        </button>
      </div>

      <button onClick={() => onSelect("watch")} style={{
        background: "transparent", border: `1px solid ${C.best}`,
        color: C.best, padding: "10px 28px", cursor: "pointer",
        fontSize: 10, letterSpacing: 3, textTransform: "uppercase",
        boxShadow: `0 0 12px ${C.best}22`,
      }}>
        Watch Agent vs Agent
      </button>

      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700;800&display=swap');`}</style>
    </div>
  );
}

// ── Game screen ───────────────────────────────────────────────────────────────
function Game({ humanPlayer, onReturnToMenu }) {
  // humanPlayer: 1 | -1 | "watch"
  const isWatch = humanPlayer === "watch";

  const [board, setBoard]       = useState(createBoard());
  const [turn, setTurn]         = useState(1);          // P1 (+1) always goes first
  const [status, setStatus]     = useState("playing");
  const [hoverCol, setHoverCol] = useState(null);
  const [agentCol, setAgentCol] = useState(null);
  const [qValues, setQValues]   = useState(null);
  const [lastDrop, setLastDrop] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [serverOk, setServerOk] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [showQ, setShowQ]       = useState(true);
  const pendingRef = useRef(false);

  useEffect(() => { checkServerHealth().then(ok => setServerOk(ok)); }, []);

  // Is it the agent's turn?
  const isAgentTurn = isWatch ? true : turn !== humanPlayer;

  // Always fetch agent hint for current position
  useEffect(() => {
    if (status !== "playing" || !serverOk) return;
    const legal = getLegal(board);
    if (!legal.length) return;
    let cancelled = false;
    (async () => {
      try {
        const { col, q_values } = await fetchAgentMove(board, legal, turn);
        if (!cancelled) { setAgentCol(col); setQValues(q_values); setApiError(null); }
      } catch (e) {
        if (!cancelled) { setApiError(e.message); setServerOk(false); }
      }
    })();
    return () => { cancelled = true; };
  }, [board, status, serverOk, turn]);

  const playMove = useCallback((col, player) => {
    const { board: nb, row } = dropPiece(board, col, player);
    setLastDrop({ row, col });
    if (checkWin(nb, player)) {
      setBoard(nb);
      // In watch mode there's no human so always "agent wins"
      setStatus(!isWatch && player === humanPlayer ? "win_human" : "win_agent");
      return;
    }
    if (!getLegal(nb).length) { setBoard(nb); setStatus("draw"); return; }
    setBoard(nb);
    setTurn(t => -t);
  }, [board, humanPlayer, isWatch]);

  // Agent auto-play (covers both sides in watch mode, only agent's side in hvsa)
  useEffect(() => {
    if (status !== "playing" || !serverOk || !isAgentTurn) return;
    if (pendingRef.current) return;
    pendingRef.current = true;
    setThinking(true);

    const legal = getLegal(board);
    const timer = setTimeout(async () => {
      try {
        const { col, q_values } = await fetchAgentMove(board, legal, turn);
        setAgentCol(col);
        setQValues(q_values);
        setApiError(null);
        playMove(col, turn);
      } catch (e) {
        setApiError(e.message);
        setServerOk(false);
      } finally {
        setThinking(false);
        pendingRef.current = false;
      }
    }, isWatch ? 700 : 450);
    return () => { clearTimeout(timer); pendingRef.current = false; };
  }, [turn, status, board, serverOk, isAgentTurn, isWatch, playMove]);

  const reset = () => {
    setBoard(createBoard()); setTurn(1); setStatus("playing");
    setHoverCol(null); setAgentCol(null); setQValues(null);
    setLastDrop(null); setThinking(false); setApiError(null);
    pendingRef.current = false;
  };

  // ── Display helpers ───────────────────────────────────────────────────────
  const canHumanPlay = !isWatch && turn === humanPlayer && status === "playing" && !thinking;
  const humanColor   = humanPlayer === 1 ? C.yellow : C.red;
  const agentColor   = humanPlayer === 1 ? C.red    : C.yellow;

  const validQ = qValues ? qValues.filter(q => q !== null) : [];
  const minQ   = validQ.length ? Math.min(...validQ) : -1;
  const maxQ   = validQ.length ? Math.max(...validQ) :  1;

  const st = (() => {
    if (status === "win_agent") return { msg: "AGENT WINS",   color: isWatch ? C.yellow : agentColor };
    if (status === "win_human") return { msg: "YOU WIN",      color: humanColor };
    if (status === "draw")      return { msg: "DRAW",         color: C.muted   };
    if (thinking)               return { msg: "THINKING…",    color: C.best    };
    if (isAgentTurn)            return { msg: "AGENT'S TURN", color: isWatch ? (turn === 1 ? C.yellow : C.red) : agentColor };
    return { msg: "YOUR TURN", color: humanColor };
  })();

  const p1Label = isWatch ? "AGENT (P1)" : humanPlayer ===  1 ? "YOU (P1)"   : "AGENT (P1)";
  const p2Label = isWatch ? "AGENT (P2)" : humanPlayer === -1 ? "YOU (P2)"   : "AGENT (P2)";

  return (
    <div style={{
      minHeight: "100vh", background: C.bg,
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", padding: 24,
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
    }}>

      {serverOk === false && (
        <div style={{ position: "fixed", top: 0, left: 0, right: 0, background: "#7f1d1d", color: "#fca5a5", textAlign: "center", padding: "10px 16px", fontSize: 12, letterSpacing: 2, zIndex: 100 }}>
          ⚠ SERVER OFFLINE — run: <code style={{ background: "#450a0a", padding: "1px 6px" }}>python serve_agent.py</code>
          {apiError && <span style={{ marginLeft: 12, opacity: 0.7 }}>{apiError}</span>}
        </div>
      )}

      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 10, letterSpacing: 8, color: C.best, marginBottom: 4 }}>DQN AGENT · LIVE</div>
        <div style={{ fontSize: 28, fontWeight: 800, color: C.text, letterSpacing: 4, textShadow: `0 0 30px ${C.best}44` }}>CONNECT FOUR</div>
        {serverOk === true && <div style={{ fontSize: 10, color: C.good, marginTop: 4, letterSpacing: 2 }}>● AGENT CONNECTED</div>}
        {serverOk === null && <div style={{ fontSize: 10, color: C.muted, marginTop: 4, letterSpacing: 2 }}>○ CONNECTING…</div>}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", justifyContent: "center" }}>
        <button onClick={() => setShowQ(q => !q)} style={{ background: showQ ? `${C.best}22` : "transparent", color: showQ ? C.best : C.muted, border: `1px solid ${showQ ? C.best : C.muted}`, padding: "5px 14px", cursor: "pointer", fontSize: 10, letterSpacing: 2, textTransform: "uppercase" }}>Q-Values {showQ ? "ON" : "OFF"}</button>
        <button onClick={reset} style={{ background: "transparent", color: C.muted, border: `1px solid ${C.muted}`, padding: "5px 14px", cursor: "pointer", fontSize: 10, letterSpacing: 2, textTransform: "uppercase" }}>Restart</button>
        <button onClick={onReturnToMenu} style={{ background: "transparent", color: C.muted, border: `1px solid ${C.muted}`, padding: "5px 14px", cursor: "pointer", fontSize: 10, letterSpacing: 2, textTransform: "uppercase" }}>← Menu</button>
      </div>

      {/* Player labels with active cursor */}
      <div style={{ display: "flex", gap: 32, marginBottom: 12, fontSize: 10, letterSpacing: 2 }}>
        {[[1, C.yellow, p1Label], [-1, C.red, p2Label]].map(([p, color, label]) => (
          <div key={p} style={{ display: "flex", alignItems: "center", gap: 6, color: turn === p && status === "playing" ? color : C.muted }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, boxShadow: turn === p && status === "playing" ? `0 0 8px ${color}` : "none" }} />
            {label}
            {turn === p && status === "playing" && <span style={{ animation: "blink 1s step-end infinite" }}>▌</span>}
          </div>
        ))}
      </div>

      {/* Status */}
      <div style={{ padding: "7px 28px", marginBottom: 14, border: `1px solid ${st.color}55`, background: `${st.color}11`, color: st.color, fontSize: 12, letterSpacing: 3, textTransform: "uppercase", boxShadow: `0 0 16px ${st.color}22`, minWidth: 220, textAlign: "center" }}>
        {st.msg}{thinking && <span style={{ marginLeft: 8, animation: "blink 0.8s step-end infinite" }}>▌</span>}
      </div>

      {/* Agent preferred move — only shown when it's the agent's turn */}
      {agentCol !== null && status === "playing" && serverOk && isAgentTurn && (
        <div style={{ fontSize: 10, color: C.best, letterSpacing: 3, marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: C.muted }}>AGENT PREFERS</span>
          <span style={{ border: `1px solid ${C.best}`, background: `${C.best}18`, padding: "2px 12px", boxShadow: `0 0 8px ${C.best}44` }}>COL {agentCol + 1}</span>
        </div>
      )}

      {/* Q bars — only shown when it's the agent's turn */}
      {showQ && qValues && status === "playing" && isAgentTurn && (
        <div style={{ display: "flex", gap: 4, marginBottom: 6, alignItems: "flex-end" }}>
          {qValues.map((q, c) => {
            const legal  = getLegal(board).includes(c);
            const isBest = c === agentCol;
            const barH   = q === null ? 0 : Math.max(4, ((q - minQ) / (maxQ - minQ + 0.001)) * 36);
            const col2   = qToColor(q, minQ, maxQ);
            return (
              <div key={c} style={{ width: 55, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                <div style={{ fontSize: 9, color: isBest ? C.best : (q !== null ? col2 : C.muted), letterSpacing: 1, height: 14, display: "flex", alignItems: "center" }}>
                  {q !== null ? (q > 0 ? `+${q.toFixed(1)}` : q.toFixed(1)) : "—"}
                </div>
                <div style={{ width: 28, height: 36, display: "flex", alignItems: "flex-end", justifyContent: "center", background: `${C.rim}44`, borderRadius: 2 }}>
                  {legal && q !== null && <div style={{ width: 28, height: barH, background: isBest ? C.best : col2, borderRadius: "2px 2px 0 0", boxShadow: isBest ? `0 0 8px ${C.best}` : "none", transition: "height 0.3s" }} />}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Drop indicator */}
      <div style={{ display: "flex", gap: 4, marginBottom: 4, height: 14 }}>
        {Array.from({ length: COLS }, (_, c) => {
          const isHover = hoverCol === c && canHumanPlay;
          const isBest  = c === agentCol && status === "playing" && serverOk && isAgentTurn;
          const legal   = getLegal(board).includes(c);
          const dotColor = isHover ? humanColor : C.best;
          return (
            <div key={c} style={{ width: 55, display: "flex", justifyContent: "center", alignItems: "center" }}>
              {legal && (isHover || isBest) && <div style={{ width: 8, height: 8, borderRadius: "50%", background: dotColor, boxShadow: `0 0 8px ${dotColor}`, animation: "drop 0.7s ease-in-out infinite alternate" }} />}
            </div>
          );
        })}
      </div>

      {/* Board */}
      <div style={{ background: C.board, border: `2px solid ${C.rim}`, borderRadius: 6, padding: 10, boxShadow: `0 0 60px ${C.rim}66, 0 20px 60px #00000088`, cursor: canHumanPlay ? "pointer" : "default" }} onMouseLeave={() => setHoverCol(null)}>
        {board.map((rowArr, row) => (
          <div key={row} style={{ display: "flex", gap: 5, marginBottom: row < ROWS - 1 ? 5 : 0 }}>
            {rowArr.map((cell, col) => {
              const isLastDrop  = lastDrop?.row === row && lastDrop?.col === col;
              const isBestCol   = agentCol === col && status === "playing" && getLegal(board).includes(col);
              const isHoverThis = hoverCol === col && canHumanPlay && cell === EMPTY;
              const { fill, glow } = pieceColor(cell);
              return (
                <div key={col} onClick={() => canHumanPlay && getLegal(board).includes(col) && playMove(col, humanPlayer)} onMouseEnter={() => setHoverCol(col)} style={{ position: "relative", width: 55, height: 55 }}>
                  <div style={{ width: 55, height: 55, borderRadius: "50%", background: fill, border: cell === EMPTY ? `1.5px solid ${isHoverThis ? humanColor + "88" : isBestCol ? C.best + "66" : C.holeRim}` : "none", boxShadow: glow ? `0 0 ${isLastDrop ? 18 : 8}px ${glow}, inset 0 -4px 10px #00000055` : "inset 0 -3px 8px #00000066", transition: "background 0.12s", position: "relative", overflow: "hidden" }}>
                    {cell !== EMPTY && <div style={{ position: "absolute", top: 9, left: 11, width: 13, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.22)", transform: "rotate(-25deg)" }} />}
                  </div>
                  {isBestCol && cell === EMPTY && serverOk && isAgentTurn && <div style={{ position: "absolute", inset: -3, borderRadius: "50%", border: `2px solid ${C.best}`, boxShadow: `0 0 12px ${C.best}88`, pointerEvents: "none", animation: "spin 4s linear infinite" }} />}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Column numbers */}
      <div style={{ display: "flex", gap: 5, marginTop: 8 }}>
        {Array.from({ length: COLS }, (_, c) => (
          <div key={c} style={{ width: 55, textAlign: "center", fontSize: 10, letterSpacing: 1, color: agentCol === c && serverOk ? C.best : C.muted }}>{c + 1}</div>
        ))}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 24, marginTop: 16, fontSize: 10, color: C.muted, letterSpacing: 2 }}>
        {[
          { dot: C.yellow, label: isWatch ? "AGENT P1" : humanPlayer === 1  ? "YOU"   : "AGENT", ring: false },
          { dot: C.red,    label: isWatch ? "AGENT P2" : humanPlayer === -1 ? "YOU"   : "AGENT", ring: false },
          { dot: C.best,   label: "PREFERRED MOVE", ring: true },
        ].map(({ dot, label, ring }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: ring ? "transparent" : dot, border: ring ? `2px solid ${dot}` : "none", boxShadow: `0 0 6px ${dot}66`, flexShrink: 0 }} />
            {label}
          </div>
        ))}
      </div>

      {/* Game over */}
      {status !== "playing" && (
        <div style={{ marginTop: 24, textAlign: "center" }}>
          <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: 5, color: status === "win_human" ? humanColor : status === "win_agent" ? (isWatch ? C.yellow : agentColor) : C.muted, textShadow: "0 0 24px currentColor", marginBottom: 14 }}>
            {status === "win_agent" ? "AGENT WINS" : status === "win_human" ? "YOU WIN" : "DRAW"}
          </div>
          <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
            <button onClick={reset} style={{ background: "transparent", color: C.best, border: `1px solid ${C.best}`, padding: "8px 28px", cursor: "pointer", fontSize: 11, letterSpacing: 4, textTransform: "uppercase", boxShadow: `0 0 16px ${C.best}44` }}>PLAY AGAIN</button>
            <button onClick={onReturnToMenu} style={{ background: "transparent", color: C.muted, border: `1px solid ${C.muted}`, padding: "8px 28px", cursor: "pointer", fontSize: 11, letterSpacing: 4, textTransform: "uppercase" }}>← MENU</button>
          </div>
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700;800&display=swap');
        @keyframes blink { 50% { opacity: 0 } }
        @keyframes drop  { from { transform: translateY(-3px) } to { transform: translateY(3px) } }
        @keyframes spin  { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
      `}</style>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function Connect4() {
  const [humanPlayer, setHumanPlayer] = useState(null);
  if (humanPlayer === null) return <SideSelect onSelect={setHumanPlayer} />;
  return <Game humanPlayer={humanPlayer} onReturnToMenu={() => setHumanPlayer(null)} />;
}
