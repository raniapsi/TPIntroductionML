"""Évalue le checkpoint retenu sur la validation, sans réentraîner le réseau."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import create_loaders
from train import MLP, ROOT

DEFAULT_CHECKPOINT = ROOT / 'checkpoints/RMSprop_l1-0.0_l2-0.0_20260930-164741-757241/best.pt'


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    targets, probabilities = [], []
    for batch in test_loader:
        probabilities.append(model(batch['features'].to(device)).cpu().numpy().reshape(-1))
        targets.append(batch['labels'].numpy().reshape(-1))
    y = np.concatenate(targets).astype(int)
    scores = np.concatenate(probabilities)
    predictions = (scores >= 0.5).astype(int)
    return {
        'n_test': len(y), 'threshold': 0.5,
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'auc': float(roc_auc_score(y, scores)),
        'confusion_matrix': confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint absent : {args.checkpoint}')
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    # Même CSV inchangé et même graine : mêmes partitions et scaler ajusté sur le train.
    _, _, test_loader = create_loaders(ROOT / 'data/cardio_train.csv', seed=config['seed'])
    if test_loader.dataset.dataset.features.shape[1] != config['input_size']:
        raise ValueError('Les variables du dataset ne correspondent pas au modèle.')
    model = MLP(config['input_size'], config['hidden_size']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    metrics = evaluate_model(model, test_loader, device)
    metrics.update(checkpoint=str(args.checkpoint), epoch=checkpoint['epoch'],
                   validation_bce=checkpoint['val_bce'], optimizer=config['optimizer'])
    output = ROOT / 'results' / args.checkpoint.parent.name / 'test_metrics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Modèle : {config["optimizer"]} | Époque retenue : {checkpoint["epoch"]} | Device : {device}')
    print(f'Exemples de test : {metrics["n_test"]} | Seuil : 0.5')
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f'{key}: {metrics[key]:.4f}')
    print('Matrice de confusion [[TN, FP], [FN, TP]] :', metrics['confusion_matrix'])
    print('Résultats sauvegardés :', output)


if __name__ == '__main__':
    main()
