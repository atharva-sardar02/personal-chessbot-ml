from flask import Flask, request, jsonify, render_template
from model_utils import load_model, predict_best_move
import json
import webbrowser
import os
import sys

app = Flask(__name__)

# 🔧 Helper to support PyInstaller .exe paths
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ✅ Load move index mapping
with open(resource_path("move_to_idx.json")) as f:
    move_to_idx = json.load(f)
idx_to_move = {str(v): k for k, v in move_to_idx.items()}

# ✅ Load hybrid CNN+LSTM model
model = load_model(resource_path("chess_cnn_model.pth"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    print(f"\n🔍 Received FEN: {fen}")

    move = predict_best_move(fen, model, move_to_idx, idx_to_move)
    print(f"🤖 Predicted Move: {move}")

    if move:
        return jsonify({"move": move})
    else:
        return jsonify({"error": "No legal move found"}), 500

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
