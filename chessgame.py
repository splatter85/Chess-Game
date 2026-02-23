import math
import random
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BOARD_SIZE = 8
TILE_SIZE = 84
PANEL_WIDTH = 230
WIDTH = BOARD_SIZE * TILE_SIZE + PANEL_WIDTH
HEIGHT = BOARD_SIZE * TILE_SIZE

WHITE = "w"
BLACK = "b"

PAWN = "p"
KNIGHT = "n"
BISHOP = "b"
ROOK = "r"
QUEEN = "q"
KING = "k"

UNICODE_PIECES = {
    "wp": "♙", "wn": "♘", "wb": "♗", "wr": "♖", "wq": "♕", "wk": "♔",
    "bp": "♟", "bn": "♞", "bb": "♝", "br": "♜", "bq": "♛", "bk": "♚",
}

PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 20000}


@dataclass
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    promotion: Optional[str] = None
    is_castle: bool = False
    is_en_passant: bool = False


@dataclass
class GameState:
    board: List[List[str]]
    side_to_move: str
    castling_rights: Dict[str, bool]
    en_passant: Optional[Tuple[int, int]]


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


class ChessEngine:
    def __init__(self) -> None:
        self.state = self._create_initial_state()

    @staticmethod
    def _create_initial_state() -> GameState:
        board = [["" for _ in range(8)] for _ in range(8)]
        board[0] = [f"b{x}" for x in [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]]
        board[1] = ["bp"] * 8
        board[6] = ["wp"] * 8
        board[7] = [f"w{x}" for x in [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]]
        return GameState(board=board, side_to_move=WHITE, castling_rights={"wk": True, "wq": True, "bk": True, "bq": True}, en_passant=None)

    def clone_state(self, state: Optional[GameState] = None) -> GameState:
        s = state if state else self.state
        return GameState(board=[row[:] for row in s.board], side_to_move=s.side_to_move, castling_rights=s.castling_rights.copy(), en_passant=s.en_passant)

    def is_square_attacked(self, row: int, col: int, by_color: str, state: Optional[GameState] = None) -> bool:
        s = state if state else self.state
        pawn_dir = -1 if by_color == WHITE else 1
        for dc in (-1, 1):
            rr, cc = row + pawn_dir, col + dc
            if in_bounds(rr, cc) and s.board[rr][cc] == f"{by_color}{PAWN}":
                return True

        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            rr, cc = row + dr, col + dc
            if in_bounds(rr, cc) and s.board[rr][cc] == f"{by_color}{KNIGHT}":
                return True

        for dr, dc, kinds in [(-1, 0, (ROOK, QUEEN)), (1, 0, (ROOK, QUEEN)), (0, -1, (ROOK, QUEEN)), (0, 1, (ROOK, QUEEN)),
                              (-1, -1, (BISHOP, QUEEN)), (-1, 1, (BISHOP, QUEEN)), (1, -1, (BISHOP, QUEEN)), (1, 1, (BISHOP, QUEEN))]:
            rr, cc = row + dr, col + dc
            while in_bounds(rr, cc):
                piece = s.board[rr][cc]
                if piece:
                    if piece[0] == by_color and piece[1] in kinds:
                        return True
                    break
                rr += dr
                cc += dc

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if in_bounds(rr, cc) and s.board[rr][cc] == f"{by_color}{KING}":
                    return True
        return False

    def is_in_check(self, color: str, state: Optional[GameState] = None) -> bool:
        s = state if state else self.state
        king = None
        for r in range(8):
            for c in range(8):
                if s.board[r][c] == f"{color}{KING}":
                    king = (r, c)
                    break
            if king:
                break
        return True if not king else self.is_square_attacked(king[0], king[1], opponent(color), s)

    def generate_pseudo_legal_moves(self, color: str, state: Optional[GameState] = None) -> List[Move]:
        s = state if state else self.state
        moves = []
        for r in range(8):
            for c in range(8):
                piece = s.board[r][c]
                if not piece or piece[0] != color:
                    continue
                p = piece[1]
                if p == PAWN:
                    self._pawn_moves(r, c, color, s, moves)
                elif p == KNIGHT:
                    for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                        rr, cc = r + dr, c + dc
                        if in_bounds(rr, cc) and (not s.board[rr][cc] or s.board[rr][cc][0] != color):
                            moves.append(Move((r, c), (rr, cc)))
                elif p in (BISHOP, ROOK, QUEEN):
                    dirs = []
                    if p in (BISHOP, QUEEN):
                        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    if p in (ROOK, QUEEN):
                        dirs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for dr, dc in dirs:
                        rr, cc = r + dr, c + dc
                        while in_bounds(rr, cc):
                            target = s.board[rr][cc]
                            if not target:
                                moves.append(Move((r, c), (rr, cc)))
                            else:
                                if target[0] != color:
                                    moves.append(Move((r, c), (rr, cc)))
                                break
                            rr += dr
                            cc += dc
                elif p == KING:
                    self._king_moves(r, c, color, s, moves)
        return moves

    def generate_legal_moves(self, color: str, state: Optional[GameState] = None) -> List[Move]:
        s = state if state else self.state
        legal = []
        for mv in self.generate_pseudo_legal_moves(color, s):
            ns = self.clone_state(s)
            self.apply_move(mv, ns)
            if not self.is_in_check(color, ns):
                legal.append(mv)
        return legal

    def apply_move(self, move: Move, state: Optional[GameState] = None) -> None:
        s = state if state else self.state
        sr, sc = move.start
        er, ec = move.end
        piece = s.board[sr][sc]
        target = s.board[er][ec]

        if piece[1] == KING:
            s.castling_rights[f"{piece[0]}k"] = False
            s.castling_rights[f"{piece[0]}q"] = False
        if piece[1] == ROOK:
            if (sr, sc) == (7, 0): s.castling_rights["wq"] = False
            if (sr, sc) == (7, 7): s.castling_rights["wk"] = False
            if (sr, sc) == (0, 0): s.castling_rights["bq"] = False
            if (sr, sc) == (0, 7): s.castling_rights["bk"] = False
        if target == "wr" and (er, ec) == (7, 0): s.castling_rights["wq"] = False
        if target == "wr" and (er, ec) == (7, 7): s.castling_rights["wk"] = False
        if target == "br" and (er, ec) == (0, 0): s.castling_rights["bq"] = False
        if target == "br" and (er, ec) == (0, 7): s.castling_rights["bk"] = False

        s.board[sr][sc] = ""
        if move.is_en_passant:
            cap_row = er + 1 if piece[0] == WHITE else er - 1
            s.board[cap_row][ec] = ""

        if move.is_castle:
            rook_from, rook_to = ((sr, 7), (sr, 5)) if ec == 6 else ((sr, 0), (sr, 3))
            s.board[rook_to[0]][rook_to[1]] = s.board[rook_from[0]][rook_from[1]]
            s.board[rook_from[0]][rook_from[1]] = ""

        s.board[er][ec] = f"{piece[0]}{move.promotion}" if move.promotion else piece
        s.en_passant = ((sr + er) // 2, sc) if piece[1] == PAWN and abs(er - sr) == 2 else None
        s.side_to_move = opponent(s.side_to_move)

    def is_checkmate(self, color: str, state: Optional[GameState] = None) -> bool:
        s = state if state else self.state
        return self.is_in_check(color, s) and not self.generate_legal_moves(color, s)

    def is_stalemate(self, color: str, state: Optional[GameState] = None) -> bool:
        s = state if state else self.state
        return not self.is_in_check(color, s) and not self.generate_legal_moves(color, s)

    def _pawn_moves(self, r: int, c: int, color: str, s: GameState, moves: List[Move]) -> None:
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        promotion_row = 0 if color == WHITE else 7
        one = r + direction
        if in_bounds(one, c) and not s.board[one][c]:
            moves.append(Move((r, c), (one, c), promotion=QUEEN if one == promotion_row else None))
            two = r + 2 * direction
            if r == start_row and in_bounds(two, c) and not s.board[two][c]:
                moves.append(Move((r, c), (two, c)))
        for dc in (-1, 1):
            rr, cc = r + direction, c + dc
            if in_bounds(rr, cc) and s.board[rr][cc] and s.board[rr][cc][0] == opponent(color):
                moves.append(Move((r, c), (rr, cc), promotion=QUEEN if rr == promotion_row else None))
        if s.en_passant:
            ep_r, ep_c = s.en_passant
            if ep_r == r + direction and abs(ep_c - c) == 1:
                moves.append(Move((r, c), (ep_r, ep_c), is_en_passant=True))

    def _king_moves(self, r: int, c: int, color: str, s: GameState, moves: List[Move]) -> None:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc) and (not s.board[rr][cc] or s.board[rr][cc][0] != color):
                    moves.append(Move((r, c), (rr, cc)))
        if self.is_in_check(color, s):
            return
        row = 7 if color == WHITE else 0
        if (r, c) == (row, 4):
            if s.castling_rights[f"{color}k"] and not s.board[row][5] and not s.board[row][6]:
                if not self.is_square_attacked(row, 5, opponent(color), s) and not self.is_square_attacked(row, 6, opponent(color), s):
                    moves.append(Move((r, c), (row, 6), is_castle=True))
            if s.castling_rights[f"{color}q"] and not s.board[row][1] and not s.board[row][2] and not s.board[row][3]:
                if not self.is_square_attacked(row, 3, opponent(color), s) and not self.is_square_attacked(row, 2, opponent(color), s):
                    moves.append(Move((r, c), (row, 2), is_castle=True))


class ChessAI:
    def __init__(self, engine: ChessEngine) -> None:
        self.engine = engine
        self.depth_by_difficulty = {"Easy": 1, "Medium": 2, "Hard": 3}
        self.blunder_chance = {"Easy": 0.4, "Medium": 0.15, "Hard": 0.02}

    def choose_move(self, difficulty: str = "Medium") -> Optional[Move]:
        state = self.engine.state
        color = state.side_to_move
        legal = self.engine.generate_legal_moves(color, state)
        if not legal:
            return None
        depth = self.depth_by_difficulty.get(difficulty, 2)
        scored = []
        for mv in legal:
            ns = self.engine.clone_state(state)
            self.engine.apply_move(mv, ns)
            scored.append((-self._search(ns, depth - 1, -math.inf, math.inf), mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        if random.random() < self.blunder_chance.get(difficulty, 0.1) and len(scored) > 1:
            return random.choice(scored[1:min(4, len(scored))])[1]
        return scored[0][1]

    def _search(self, state: GameState, depth: int, alpha: float, beta: float) -> int:
        color = state.side_to_move
        if self.engine.is_checkmate(color, state):
            return -999999 + (3 - depth)
        if self.engine.is_stalemate(color, state):
            return 0
        if depth == 0:
            return self._eval(state)
        best = -math.inf
        for mv in self.engine.generate_legal_moves(color, state):
            ns = self.engine.clone_state(state)
            self.engine.apply_move(mv, ns)
            score = -self._search(ns, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return int(best)

    def _eval(self, state: GameState) -> int:
        score = 0
        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece:
                    score += PIECE_VALUES[piece[1]] if piece[0] == state.side_to_move else -PIECE_VALUES[piece[1]]
        return score


class ChessTkApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Chess vs Computer (Python 3.14 friendly)")

        self.engine = ChessEngine()
        self.ai = ChessAI(self.engine)
        self.human_color = WHITE
        self.ai_color = BLACK

        self.drag_from: Optional[Tuple[int, int]] = None
        self.drag_piece: Optional[str] = None
        self.drag_visual = None
        self.legal_for_selected: List[Move] = []

        self.canvas = tk.Canvas(self.root, width=BOARD_SIZE * TILE_SIZE, height=HEIGHT, bg="#222")
        self.canvas.grid(row=0, column=0)

        panel = tk.Frame(self.root, width=PANEL_WIDTH, height=HEIGHT, bg="#2b2f33")
        panel.grid(row=0, column=1, sticky="ns")

        tk.Label(panel, text="Chess AI", font=("Arial", 18, "bold"), bg="#2b2f33", fg="white").pack(pady=(14, 8))
        tk.Label(panel, text="Difficulty", font=("Arial", 11), bg="#2b2f33", fg="#d8d8d8").pack()

        self.difficulty = tk.StringVar(value="Medium")
        tk.OptionMenu(panel, self.difficulty, "Easy", "Medium", "Hard").pack(pady=8)

        tk.Button(panel, text="New Game", width=18, command=self.new_game).pack(pady=6)
        tk.Button(panel, text="Switch Side", width=18, command=self.switch_side).pack(pady=6)

        self.turn_var = tk.StringVar()
        self.side_var = tk.StringVar()
        self.msg_var = tk.StringVar()
        tk.Label(panel, textvariable=self.turn_var, bg="#2b2f33", fg="#e5e5e5", font=("Arial", 11)).pack(pady=(30, 4))
        tk.Label(panel, textvariable=self.side_var, bg="#2b2f33", fg="#e5e5e5", font=("Arial", 11)).pack(pady=(0, 20))
        tk.Label(panel, textvariable=self.msg_var, bg="#2b2f33", fg="#ffd67f", font=("Arial", 11), wraplength=190, justify="left").pack(padx=10)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.draw()

    def square_from_xy(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if not (0 <= x < BOARD_SIZE * TILE_SIZE and 0 <= y < HEIGHT):
            return None
        return y // TILE_SIZE, x // TILE_SIZE

    def on_mouse_down(self, event) -> None:
        if self.engine.state.side_to_move != self.human_color:
            return
        sq = self.square_from_xy(event.x, event.y)
        if not sq:
            return
        r, c = sq
        piece = self.engine.state.board[r][c]
        if piece and piece[0] == self.human_color:
            self.drag_from = sq
            self.drag_piece = piece
            self.legal_for_selected = [m for m in self.engine.generate_legal_moves(self.human_color) if m.start == sq]
            self.draw()

    def on_mouse_drag(self, event) -> None:
        if not self.drag_piece:
            return
        self.draw()
        self.drag_visual = self.canvas.create_text(event.x, event.y, text=UNICODE_PIECES[self.drag_piece], font=("DejaVu Sans", 46), fill="#111")

    def on_mouse_up(self, event) -> None:
        if not self.drag_from:
            return
        target = self.square_from_xy(event.x, event.y)
        moved = False
        if target:
            for mv in self.legal_for_selected:
                if mv.end == target:
                    self.engine.apply_move(mv)
                    moved = True
                    break
        self.drag_from = None
        self.drag_piece = None
        self.legal_for_selected = []
        self.draw()
        if moved:
            self.update_status()
            self.root.after(120, self.maybe_ai_move)

    def maybe_ai_move(self) -> None:
        if self.engine.state.side_to_move != self.ai_color:
            return
        if self.engine.is_checkmate(self.ai_color) or self.engine.is_stalemate(self.ai_color):
            return
        mv = self.ai.choose_move(self.difficulty.get())
        if mv:
            self.engine.apply_move(mv)
            self.draw()
            self.update_status()

    def update_status(self) -> None:
        stm = self.engine.state.side_to_move
        if self.engine.is_checkmate(stm):
            self.msg_var.set(f"Checkmate! {'White' if opponent(stm) == WHITE else 'Black'} wins")
        elif self.engine.is_stalemate(stm):
            self.msg_var.set("Draw by stalemate")
        elif self.engine.is_in_check(stm):
            self.msg_var.set(f"{'White' if stm == WHITE else 'Black'} is in check")
        else:
            self.msg_var.set("")
        self.turn_var.set(f"Turn: {'White' if stm == WHITE else 'Black'}")
        self.side_var.set(f"You: {'White' if self.human_color == WHITE else 'Black'}")

    def draw(self) -> None:
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                color = "#ecdbae" if (r + c) % 2 == 0 else "#946a42"
                x0, y0 = c * TILE_SIZE, r * TILE_SIZE
                self.canvas.create_rectangle(x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE, fill=color, outline=color)

        if self.drag_from:
            r, c = self.drag_from
            x0, y0 = c * TILE_SIZE, r * TILE_SIZE
            self.canvas.create_rectangle(x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE, outline="#59afff", width=3)

        for mv in self.legal_for_selected:
            r, c = mv.end
            cx = c * TILE_SIZE + TILE_SIZE // 2
            cy = r * TILE_SIZE + TILE_SIZE // 2
            self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#333", outline="")

        for r in range(8):
            for c in range(8):
                piece = self.engine.state.board[r][c]
                if not piece:
                    continue
                if self.drag_from == (r, c) and self.drag_piece:
                    continue
                cx = c * TILE_SIZE + TILE_SIZE // 2
                cy = r * TILE_SIZE + TILE_SIZE // 2 + 3
                color = "#f0f0f0" if piece[0] == BLACK else "#222"
                self.canvas.create_text(cx, cy, text=UNICODE_PIECES[piece], font=("DejaVu Sans", 46), fill=color)

        self.update_status()

    def new_game(self) -> None:
        self.engine = ChessEngine()
        self.ai = ChessAI(self.engine)
        self.drag_from = None
        self.drag_piece = None
        self.legal_for_selected = []
        self.draw()

    def switch_side(self) -> None:
        self.human_color, self.ai_color = self.ai_color, self.human_color
        self.new_game()
        if self.engine.state.side_to_move == self.ai_color:
            self.root.after(120, self.maybe_ai_move)

    def run(self) -> None:
        if self.engine.state.side_to_move == self.ai_color:
            self.root.after(120, self.maybe_ai_move)
        self.root.mainloop()


if __name__ == "__main__":
    ChessTkApp().run()
