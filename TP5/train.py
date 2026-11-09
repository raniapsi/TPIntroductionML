import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader, test_loader
from model import ImageCaptioningModel, resnet_pretrained
from text_utils import (
    pretrained_embeddings,
    word2idx,
    idx2word,
    tokenize,
    untokenize,
)


def main():
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Appareil utilisé :", device)

    model = ImageCaptioningModel(
        cnn_model=resnet_pretrained,
        embedding_model=pretrained_embeddings,
        lstm_hidden_dim=512,
    ).to(device)

    for param in model.cnn.parameters():
        param.requires_grad = False

    model.embeddings.weight.requires_grad = False

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.001,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1,
    )
    writer = SummaryWriter("runs/captioning")

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        model.cnn.eval()
        total_loss = 0
        epoch_loss = 0

        print(
            f"\nÉpoque {epoch + 1}/{num_epochs} | "
            f"LR utilisé : {optimizer.param_groups[0]['lr']}"
        )

        for i, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)

            # Tokenisation manuelle du batch (simplifiée)
            caps = torch.tensor(
                [tokenize(x, word2idx) for x in captions[0]],
                dtype=torch.long,
                device=device,
            )

            # On fournit toutes les séquences sauf le dernier mot en entrée
            outputs = model(images, caps[:, :-1])

            # On s'attend à prédire la séquence décalée d'un cran
            # Reshape pour la Loss : (batch_size * seq_len, vocab_size)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                writer.add_scalar(
                    "Loss/Train_Step",
                    total_loss / 10,
                    epoch * len(train_loader) + i,
                )
                print(
                    f"Batch {i + 1}/{len(train_loader)} | "
                    f"Loss : {total_loss / 10:.4f}",
                    flush=True,
                )
                total_loss = 0

        # Évaluation et génération
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, captions in test_loader:
                images = images.to(device)
                caps = torch.tensor(
                    [tokenize(x, word2idx) for x in captions[0]],
                    dtype=torch.long,
                    device=device,
                )

                outputs = model(images, caps[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    caps[:, 1:].reshape(-1),
                )
                val_loss += loss.item()

                # Visualisation d'une prédiction
                predictions = outputs[-1].argmax(dim=-1)
                pred_text = " ".join(
                    untokenize(predictions.cpu().tolist(), idx2word)
                )
                true_text = " ".join(
                    untokenize(caps[-1][1:].cpu().tolist(), idx2word)
                )

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss : {epoch_loss / len(train_loader):.4f}")
        print(f"Val Loss   : {val_loss / len(test_loader):.4f}")
        print(f"Prediction : {pred_text}")
        print(f"Target     : {true_text}", flush=True)

        writer.add_scalar(
            "Loss/Validation",
            val_loss / len(test_loader),
            epoch,
        )

        scheduler.step()
        writer.add_scalar(
            "LR",
            scheduler.get_last_lr()[0],
            epoch,
        )
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
