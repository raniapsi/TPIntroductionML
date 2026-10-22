import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ---------- Configuration ----------

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------- Préparation des images ----------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...", flush=True)

    train_dataset = ImageNet(
        root=data_dir,
        split="train",
        transform=train_transform,
    )

    val_dataset = ImageNet(
        root=data_dir,
        split="val",
        transform=val_transform,
    )

    if subset_size is not None:
        if subset_size < 4:
            raise ValueError("subset_size doit être au moins égal à 4.")

        generator = torch.Generator().manual_seed(42)

        train_indices = torch.randperm(
            len(train_dataset), generator=generator
        )[:subset_size]

        val_indices = torch.randperm(
            len(val_dataset), generator=generator
        )[:subset_size // 4]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    print(
        f"Images d'entraînement : {len(train_dataset)} | "
        f"Images de validation : {len(val_dataset)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader


# ---------- MLP ----------

class MLP(nn.Module):
    def __init__(
        self,
        input_size=3 * 227 * 227,
        hidden_size=256,
        num_classes=1000,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------- AlexNet standard ----------

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Dernier pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------- AlexNet profond ----------

class AlexNetDeep(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes=num_classes)

        layers = list(self.features.children())

        self.features = nn.Sequential(
            *layers[:-1],

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            layers[-1],
        )


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


# ---------- Évaluation ----------

@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ---------- Entraînement ----------

def train_model(
    model_name="alexnet",
    batch_size=256,
    lr=1e-4,
    num_epochs=20,
    weight_decay=1e-5,
    patience=5,
    subset_size=20000,
):
    torch.manual_seed(42)

    n_workers = min(
        8,
        max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))),
    )

    print(f"Appareil utilisé : {device}", flush=True)
    print(f"Workers pour le chargement : {n_workers}", flush=True)

    if device.type == "cuda":
        print(
            f"GPU : {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        n_workers=n_workers,
        subset_size=subset_size,
    )

    if model_name == "alexnet":
        model = AlexNet(num_classes=1000).to(device)
    elif model_name == "alexnet_deep":
        model = AlexNetDeep(num_classes=1000).to(device)
    elif model_name == "mlp":
        model = MLP(num_classes=1000).to(device)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    print(
        f"Modèle {model_name} : "
        f"{count_parameters(model):,} paramètres.",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dataset_tag = "full" if subset_size is None else f"sub{subset_size}"
    run_tag = f"{model_name}_{dataset_tag}_lr{lr}_{timestamp}"
    log_dir = f"runs/{run_tag}"

    best_val_loss = float("inf")
    early_stop_counter = 0

    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(num_epochs):
            # Début du chronométrage
            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_start = time.perf_counter()

            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                mininterval=10,
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            # Durée de l'entraînement seul
            if device.type == "cuda":
                torch.cuda.synchronize()

            train_seconds = time.perf_counter() - epoch_start

            train_loss = running_loss / total_train
            train_acc = correct_train / total_train

            val_loss, val_acc = evaluate(
                val_loader, model, criterion
            )

            # Durée de l'entraînement + validation
            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_seconds = time.perf_counter() - epoch_start

            writer.add_scalar(
                "Loss/Train", train_loss, epoch + 1
            )
            writer.add_scalar(
                "Loss/Validation", val_loss, epoch + 1
            )
            writer.add_scalar(
                "Accuracy/Train", train_acc, epoch + 1
            )
            writer.add_scalar(
                "Accuracy/Validation", val_acc, epoch + 1
            )
            writer.add_scalar(
                "Time/Train_seconds", train_seconds, epoch + 1
            )
            writer.add_scalar(
                "Time/Epoch_seconds", epoch_seconds, epoch + 1
            )
            writer.flush()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} "
                f"(Acc: {train_acc:.4f}) | "
                f"Val Loss: {val_loss:.4f} "
                f"(Acc: {val_acc:.4f})",
                flush=True,
            )

            print(
                f"Durée entraînement seul : {train_seconds:.1f} s | "
                f"Durée entraînement + validation : "
                f"{epoch_seconds:.1f} s "
                f"({epoch_seconds / 60:.2f} min)",
                flush=True,
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0

                # Sauvegarde des meilleurs poids
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, "best_model.pth"),
                )
            else:
                early_stop_counter += 1

                if early_stop_counter >= patience:
                    print("Early stopping déclenché !", flush=True)
                    break

    print(f"Entraînement terminé. Courbes : {log_dir}", flush=True)


# ---------- Exercice 4 : entraînement complet ----------

if __name__ == "__main__":
    train_model(
        model_name="alexnet",
        subset_size=None,
        num_epochs=50,
    )







































