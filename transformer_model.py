import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ======================================================
# Dataset
# ======================================================

class TumorProgressionDataset(Dataset):
    def __init__(self, seq_path, target_path):
        self.X = np.load(seq_path)
        self.y = np.load(target_path)

        assert self.X.ndim == 3, "X must be (N, T, F)"
        assert self.y.shape[1] == 3, "Targets must be (death, survival, recurrence)"

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================================================
# Positional Encoding
# ======================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ======================================================
# Transformer Model
# ======================================================

class TumorProgressionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_model)

        # Heads
        self.death_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.survival_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.recurrence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        encoded = self.encoder(x)

        pooled = encoded.mean(dim=1)
        pooled = self.norm(pooled)

        death = self.death_head(pooled)
        survival = self.survival_head(pooled)
        recurrence = self.recurrence_head(pooled)

        return torch.cat([death, survival, recurrence], dim=1)


# ======================================================
# Multi-task Loss
# ======================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, w_death=1.0, w_survival=0.5, w_recurrence=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.w_death = w_death
        self.w_survival = w_survival
        self.w_recurrence = w_recurrence

    def forward(self, preds, targets):
        ld = self.bce(preds[:, 0], targets[:, 0])
        ls = self.mse(preds[:, 1], targets[:, 1])
        lr = self.bce(preds[:, 2], targets[:, 2])

        total = (
            self.w_death * ld
            + self.w_survival * ls
            + self.w_recurrence * lr
        )

        return total, {
            "death": ld.item(),
            "survival": ls.item(),
            "recurrence": lr.item()
        }


# ======================================================
# Training Loop
# ======================================================

def train(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epochs,
    save_path
):
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)

            loss, loss_dict = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                d=f"{loss_dict['death']:.3f}",
                s=f"{loss_dict['survival']:.3f}",
                r=f"{loss_dict['recurrence']:.3f}"
            )

        avg_loss = epoch_loss / len(loader)
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {avg_loss:.4f}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Model saved to: {save_path}")


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    SEQ_PATH = r"K:\HackRush\synthetic_tumor_sequences.npy"
    TARGET_PATH = r"K:\HackRush\synthetic_targets.npy"
    MODEL_SAVE_PATH = r"K:\HackRush\models\tumor_progression_transformer.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TumorProgressionDataset(SEQ_PATH, TARGET_PATH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    _, T, F = dataset.X.shape

    model = TumorProgressionTransformer(
        feature_dim=F,
        d_model=128,
        n_heads=4,
        n_layers=4
    ).to(device)

    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=30,
        save_path=MODEL_SAVE_PATH
    )
