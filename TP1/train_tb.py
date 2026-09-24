import os
import random
import datetime
import torch

from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets


# Hyperparamètres de cet entraînement
hparams = {
    "model": "MLP",
    "batch_size": 128,
    "lr": 1e-1,
    "seed": 0,
    "weight_decay": 0.0,
}

# Nom unique pour retrouver cet entraînement dans TensorBoard
run_name = (
    f"{hparams['model']}/"
    f"bs{hparams['batch_size']}_"
    f"lr{hparams['lr']}_"
    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

logdir = os.path.join("runs", run_name)
print("Logdir:", logdir)

writer = SummaryWriter(log_dir=logdir)


# Reproductibilité
torch.manual_seed(hparams["seed"])
random.seed(hparams["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams["seed"])


# Préparation de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)


# Séparation : 90 % entraînement et 10 % validation
N = len(trainset)
val_size = int(0.1 * N)
train_size = N - val_size

train_subset, val_subset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(hparams["seed"]),
)


# Respecter le nombre de CPU attribué par Slurm
num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))

trainloader = DataLoader(
    train_subset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

valloader = DataLoader(
    val_subset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 × 32 × 32 = 3072 valeurs par image
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        # Additionner les pertes de tous les exemples
        loss_sum += loss.item() * y.size(0)

        # Compter les bonnes prédictions
        predictions = logits.argmax(dim=1)
        correct += (predictions == y).sum().item()
        total += y.size(0)

    loss_avg = loss_sum / total
    accuracy = correct / total

    return loss_avg, accuracy
# Préparer le modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hparams["lr"],
    momentum=0.9,
    weight_decay=hparams["weight_decay"],
)


# Entraînement
global_step = 0
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss_sum = 0.0
    running_total = 0

    for batch_number, (x, y) in enumerate(trainloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()

        # Enregistrer la perte d’un batch sur dix
        if batch_number % 10 == 0:
            writer.add_scalar(
                "Loss/train_step",
                loss.item(),
                global_step,
            )

        optimizer.step()

        running_loss_sum += loss.item() * y.size(0)
        running_total += y.size(0)
        global_step += 1

    # Perte moyenne sur toutes les images d’entraînement
    train_loss = running_loss_sum / running_total

    # Résultats sur les images de validation
    val_loss, val_acc = epoch_metrics(
        valloader,
        model,
        criterion,
        device,
    )

    # Enregistrer les résultats de l’époque
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.3f}"
    )


# Forcer l’écriture des données puis fermer TensorBoard
writer.flush()
writer.close()

print("Entraînement terminé")
print("Logs enregistrés dans :", logdir)
