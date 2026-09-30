"""Premier exercice : dataset cardiovasculaire et DataLoaders sans data leakage."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split


class CardioDataset(Dataset):
    def __init__(self, csv_path):
        frame = pd.read_csv(csv_path, sep=";").drop_duplicates().drop(columns="id")
        self.labels = frame.pop("cardio").to_numpy(dtype=np.float32).reshape(-1, 1)
        frame = pd.get_dummies(frame, columns=["gender", "cholesterol", "gluc"], dtype=np.float32)
        self.feature_names = list(frame.columns)
        # Le scaler sera ajusté après le découpage, sur le train uniquement.
        self.features = frame.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"features": self.features[idx], "labels": self.labels[idx]}


def create_loaders(csv_path, batch_size=64, seed=42):
    dataset = CardioDataset(csv_path)
    split_generator = torch.Generator().manual_seed(seed)
    train, val, test = random_split(dataset, [0.8, 0.1, 0.1], generator=split_generator)
    scaler = StandardScaler().fit(dataset.features[train.indices])
    dataset.features = scaler.transform(dataset.features).astype(np.float32)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "data" / "cardio_train.csv")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Fichier manquant : {args.csv}. Télécharger cardio_train.csv depuis Kaggle.")
    loaders = create_loaders(args.csv)
    print("Tailles train / validation / test :", [len(loader.dataset) for loader in loaders])
    batch = next(iter(loaders[0]))
    print("Shape features:", batch["features"].shape, "Shape labels:", batch["labels"].shape)
    print("Nombre de variables après encodage :", batch["features"].shape[1])
