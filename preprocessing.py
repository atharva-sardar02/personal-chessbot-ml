import json
import chess
import numpy as np
from tqdm import tqdm
import torch

# Load JSON
with open("my_chess_games.json") as f:
    games = json.load(f)

# Create a fixed list of all possible legal UCI moves (4672 max in chess)
uci_moves = [move.uci() for move in chess.Board().legal_moves]
all_possible_moves = set()

# Helper: Encode board as (8, 8, 12) tensor
def board_to_tensor(board):
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = square % 8
            y = 7 - square // 8
            idx = piece_map[piece.symbol()]
            tensor[y, x, idx] = 1
    return tensor

# Step 1: Collect all UCI moves used in your games
for game in games:
    board = chess.Board()
    moves = game["moves"]
    for move in moves:
        try:
            uci = board.parse_san(move).uci()
            all_possible_moves.add(uci)
            board.push_uci(uci)
        except:
            break  # skip malformed moves

# Build move-to-index and index-to-move dictionaries
uci_list = sorted(list(all_possible_moves))
move_to_idx = {uci: i for i, uci in enumerate(uci_list)}
idx_to_move = {i: uci for uci, i in move_to_idx.items()}

# Step 2: Generate training examples
X, y = [], []

for game in tqdm(games):
    board = chess.Board()
    player_color = "white" if game["white"] == "jarvisnoble" else "black"
    is_my_turn = board.turn == (player_color == "white")

    for move in game["moves"]:
        try:
            uci = board.parse_san(move).uci()
            if is_my_turn:
                board_tensor = board_to_tensor(board)
                move_idx = move_to_idx[uci]
                X.append(board_tensor)
                y.append(move_idx)
            board.push_uci(uci)
            is_my_turn = not is_my_turn
        except:
            break

# Convert to PyTorch tensors
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Save preprocessed data
torch.save((X, y, move_to_idx), "behavior_cloning_data.pt")
