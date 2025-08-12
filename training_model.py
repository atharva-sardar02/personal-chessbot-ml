import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import json

# Load preprocessed data
X, y, move_to_idx = torch.load("behavior_cloning_data.pt")
print(f"Total samples: {len(X)}, Unique moves: {len(move_to_idx)}")

# Dataset and split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# CNN+LSTM Model
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
        x = x.permute(0, 3, 1, 2)  # (B, 12, 8, 8)
        x = self.cnn(x)            # (B, 64, 8, 8)
        x = x.permute(0, 2, 3, 1)  # (B, 8, 8, 64)
        x = x.reshape(x.size(0), 8, -1)  # (B, 8, 512)
        x, _ = self.lstm(x)        # (B, 8, 128)
        x = x.reshape(x.size(0), -1)  # (B, 8*128)
        return self.fc(x)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessCNNLSTM(num_moves=len(move_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Train
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (preds.argmax(1) == yb).sum().item()

    train_acc = 100 * total_correct / len(train_set)

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            val_correct += (preds.argmax(1) == yb).sum().item()

    val_acc = 100 * val_correct / len(val_set)
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# Save
torch.save(model.state_dict(), "chess_cnn_model.pth")
with open("move_to_idx.json", "w") as f:
    json.dump(move_to_idx, f)

print("✅ Model and mapping saved!")
