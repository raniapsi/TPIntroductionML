"""Entraînement du MLP avec régularisation L1/L2 explicite."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import create_loaders

ROOT = Path(__file__).resolve().parent


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def measure(model, loader, criterion, device):
    """Mesure la BCE sans pénalités et l'accuracy sur un ensemble complet."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for batch in loader:
        x, y = batch['features'].to(device), batch['labels'].to(device)
        probabilities = model(x)
        loss_sum += criterion(probabilities, y).item() * len(y)
        correct += ((probabilities >= 0.5) == (y >= 0.5)).sum().item()
        count += len(y)
    return loss_sum / count, correct / count


def train_model(l1_lambda=1e-4, l2_lambda=1e-3, epochs=10, lr=0.01, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = create_loaders(ROOT / 'data/cardio_train.csv', seed=seed)
    input_size = train_loader.dataset.dataset.features.shape[1]
    model = MLP(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    run_name = f'SGD_l1-{l1_lambda}_l2-{l2_lambda}_{datetime.now():%Y%m%d-%H%M%S-%f}'
    result_dir = ROOT / 'results' / run_name
    result_dir.mkdir(parents=True)
    print(f'Device : {device} | Entrées : {input_size} | L1 : {l1_lambda} | L2 : {l2_lambda}', flush=True)
    print(f'Résultats : {result_dir}', flush=True)
    fields = ['epoch', 'objective', 'train_bce', 'train_accuracy', 'val_bce', 'val_accuracy']
    with SummaryWriter(str(ROOT / 'runs' / run_name)) as writer, (result_dir / 'history.csv').open('w', newline='') as output:
        table = csv.DictWriter(output, fieldnames=fields)
        table.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            objective_sum, count = 0.0, 0
            for batch in train_loader:
                x, y = batch['features'].to(device), batch['labels'].to(device)
                optimizer.zero_grad()
                base_loss = criterion(model(x), y)
                # Comme dans l'énoncé, les pénalités incluent poids et biais.
                l1 = sum(p.abs().sum() for p in model.parameters())
                l2 = sum(p.square().sum() for p in model.parameters())
                loss = base_loss + l1_lambda * l1 + l2_lambda * l2
                loss.backward()
                optimizer.step()
                objective_sum += loss.item() * len(y)
                count += len(y)
            train_bce, train_accuracy = measure(model, train_loader, criterion, device)
            val_bce, val_accuracy = measure(model, val_loader, criterion, device)
            row = dict(epoch=epoch, objective=objective_sum / count,
                       train_bce=train_bce, train_accuracy=train_accuracy,
                       val_bce=val_bce, val_accuracy=val_accuracy)
            table.writerow(row)
            output.flush()
            for key, value in row.items():
                if key != 'epoch':
                    writer.add_scalar(key, value, epoch)
            print(f'Epoch {epoch:02d}/{epochs} | objectif={row["objective"]:.4f} '
                  f'| train BCE={train_bce:.4f} acc={train_accuracy:.4f} '
                  f'| val BCE={val_bce:.4f} acc={val_accuracy:.4f}', flush=True)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l1', type=float, default=1e-4)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.l1 < 0 or args.l2 < 0:
        parser.error('epochs et lr doivent être positifs ; l1 et l2 doivent être positifs ou nuls.')
    train_model(args.l1, args.l2, args.epochs, args.lr)
