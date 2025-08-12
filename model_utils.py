import torch
import torch.nn as nn
import chess
import numpy as np
import os
import sys

class ChessCNNLSTM(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64 * 8, hidden_size=128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_moves)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, 12, 8, 8]
        x = self.cnn(x)           # [B, 64, 8, 8]
        x = x.permute(0, 2, 3, 1) # [B, 8, 8, 64]
        x = x.reshape(x.size(0), 8, -1)  # [B, 8, 512]
        x, _ = self.lstm(x)       # [B, 8, 128]
        x = x.reshape(x.size(0), -1)  # [B, 1024]
        return self.fc(x)

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def board_to_tensor(board):
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = square % 8
            y = 7 - square // 8
            idx = piece_map[piece.symbol()]
            tensor[y, x, idx] = 1
    return tensor

def load_model(path=None):
    if path is None:
        path = resource_path("chess_cnn_model.pth")
    model = ChessCNNLSTM(num_moves=1653)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_best_move(fen, model, move_to_idx, idx_to_move):
    board = chess.Board(fen)
    state = torch.tensor(board_to_tensor(board)).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)
        probs = torch.softmax(logits, dim=1)
        sorted_indices = torch.argsort(probs, descending=True).squeeze()

        for idx in sorted_indices:
            move_str = idx_to_move[str(idx.item())]
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move.uci()
    return None
