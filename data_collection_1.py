import requests
import chess.pgn
import io
import json

# ✅ Your Chess.com username
username = "jarvisnoble"

# Step 1: Get list of monthly archive URLs
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

archive_url = f"https://api.chess.com/pub/player/{username}/games/archives"
response = requests.get(archive_url, headers=headers)

print(f"Status code: {response.status_code}")
print(f"Response preview:\n{response.text[:300]}")

if response.status_code == 200:
    archives = response.json()["archives"]
else:
    print("❌ Still blocked. Try disabling VPN or wait and retry.")
    exit()

all_games_data = []

# Step 2: Fetch and parse PGN for each month
for archive in archives:
    print(f"📥 Fetching: {archive}")
    res = requests.get(archive, headers=headers)
    if res.status_code != 200:
        print(f"❌ Failed to fetch: {archive}")
        continue

    games = res.json()["games"]
    for game in games:
        pgn_text = game.get("pgn", "")
        game_io = io.StringIO(pgn_text)
        parsed_game = chess.pgn.read_game(game_io)

        if parsed_game is None:
            continue

        # Extract moves in SAN notation
        moves = []
        board = parsed_game.board()
        for move in parsed_game.mainline_moves():
            san_move = board.san(move)  # get the SAN before pushing
            board.push(move)
            moves.append(san_move)

        # Extract useful metadata
        game_data = {
            "white": parsed_game.headers.get("White"),
            "black": parsed_game.headers.get("Black"),
            "result": parsed_game.headers.get("Result"),
            "end_time": parsed_game.headers.get("EndTime"),
            "time_control": parsed_game.headers.get("TimeControl"),
            "termination": parsed_game.headers.get("Termination"),
            "moves": moves
        }

        all_games_data.append(game_data)

print(f"\n✅ Total games fetched: {len(all_games_data)}")

# Step 3: Save all data to a JSON file
with open("my_chess_games.json", "w") as f:
    json.dump(all_games_data, f, indent=2)

print("✅ Data saved to: my_chess_games.json")