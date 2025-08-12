# ♟️ Chess Bot – Behavior Cloning (CNN → CNN+LSTM) & Flask Web UI

A personalized chess engine trained on my own games using **Behavior Cloning** with **Convolutional Neural Networks (CNN)** and an optional **CNN+LSTM** architecture for sequence-based decision-making. Integrated with a **Flask** backend and **chessboard.js** UI for interactive play directly in the browser.

---

## 🚀 Features

- **Behavior Cloning from Personal Games**
  - Extracts `(board → move)` pairs from PGN files.
  - Trains a supervised learning model to mimic my playstyle.
- **CNN Model**
  - Encodes board state into feature maps for move prediction.
- **Optional CNN+LSTM**
  - Learns sequences of board positions for context-aware predictions.
- **Flask Web App**
  - Play against the trained bot in the browser.
  - Side selection (White/Black), undo, restart, legal-move highlighting.
- **Move Filtering**
  - Ensures only legal moves are returned by the model.
- **Model Export/Reload**
  - Save weights as `.pth` and load for inference.

---

## 🛠 Tech Stack

**Core ML**
- Python, PyTorch, NumPy, Pandas, Scikit-Learn

**Backend & UI**
- Flask, chess.js, chessboard.js, HTML/CSS/JavaScript

**Data Handling**
- python-chess (FEN & PGN parsing)
- JSON-based move-index mapping

---

## 📂 Repository Structure

```plaintext
├── build/                           # Build files (if packaged)
├── dist/
│   └── app.exe                       # Packaged executable (optional)
├── static/
│   └── chessboard.min.js             # Chessboard UI JS
├── templates/
│   └── index.html                    # Web UI
├── app.py                            # Flask backend
├── app.spec                          # PyInstaller spec (optional)
├── behavior_cloning_data.pt          # Preprocessed dataset (tensor format)
├── chess_cnn_model.pth               # Trained CNN model weights
├── data_collection_1.py              # PGN → board-move pair extraction
├── figenbaumgraph.py                 # Utility/analysis script
├── model_utils.py                    # Model architectures & helpers
├── move_to_idx.json                  # Mapping (move → index)
├── my_chess_games.json               # Processed games dataset
├── preprocessing.py                  # Data preprocessing scripts
├── training_model.py                 # Training pipeline (CNN / CNN+LSTM)
└── README.md                         # Project documentation
```
## 📦 Data Prep

Provide your games as PGN files or a prebuilt JSON of `(board_state → move)` pairs.  
This project supports **FEN sequences** for CNN+LSTM training and single-board encoding for CNN.

**Typical workflow (example):**
```bash
# 1️⃣ Convert PGNs to supervised pairs
# (Adjust script paths and file names as needed)
python data_collection_1.py --pgn ./data/my_games.pgn --out ./my_chess_games.json
```
```bash
# 2️⃣ (Optional) Cache tensorized dataset for faster training
python preprocessing.py --in ./my_chess_games.json --out ./behavior_cloning_data.pt
```
 **Note:**
 move_to_idx.json gets created/updated during preprocessing/training.

## 🧠 Training

**Train the CNN Baseline**
```bash
python training_model.py \
  --data ./behavior_cloning_data.pt \
  --epochs 15 \
  --batch-size 256 \
  --lr 1e-3 \
  --save ./chess_cnn_model.pth
```
**Enable CNN+LSTM Training**
```bash
python training_model.py \
  --data ./behavior_cloning_data.pt \
  --sequence-len 8 \
  --model cnn_lstm \
  --epochs 20 \
  --batch-size 128 \
  --lr 5e-4 \
  --save ./chess_cnn_lstm_model.pth
```
>**Note:**

- The --sequence-len flag expects sequences of positions per sample.

- Ensure your preprocessing emits sequences when using --model cnn_lstm.

## 🕹️ Run the Web App

```bash
# Default: loads ./chess_cnn_model.pth
python app.py --model ./chess_cnn_model.pth --move-index ./move_to_idx.json
```
> **Note:** Then open the Flask URL in your browser (usually: http://127.0.0.1:5000/).

## UI Features

- Choose side: **White** / **Black**
- Show legal moves
- **Predict** button to get the model’s move
- Undo / Restart game

---

## 🔌 Minimal API (Backend)

**`POST /predict`**
```json
{"fen": "<FEN string>"}
```
**Returns**
```json
{"uci": "e2e4", "san": "e4", "score": 0.73}
```
**`POST /reset`**
- Resets the server-side board/session (if applicable).

> **Note:** The Flask backend automatically filters predictions to **legal moves only** before returning.

## 📈 Tips for Better Play

- Balance your dataset: avoid only short wins or losses.
- Increase variety: include puzzles and tactics in training data.
- For sequence training: use CNN+LSTM with `--sequence-len >= 6`.
- Apply regularization: weight decay, dropout; validate on held-out games.
- Calibrate model predictions: softmax temperature scaling or top-k sampling in `app.py`.

---

## 🧪 Repro & Packaging

**Freeze exact environment** *(optional)*:
```bash
pip freeze > requirements-lock.txt
```
**Create a single-file executable (optional, Windows):**
```bash
pip install pyinstaller
pyinstaller --onefile --name app app.spec
# or:
pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" app.py
```
