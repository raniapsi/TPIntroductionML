# TP2 — Régularisation, optimisation et métriques

Travail progressif sur le cluster, dans le même dépôt que le TP1.

- [Énoncé](https://www-inf.telecom-sudparis.eu/COURS/CSC8607/Supports/?page=exercices/ci2&wrap=true)
- [Cours](https://www-inf.telecom-sudparis.eu/COURS/CSC8607/Supports/cours/regularisation_optimisation.html)
- [Dataset officiel Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Première exécution

Dans une session de calcul du cluster, à la racine du dépôt :

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
cd TP2
python -m pip install -r requirements.txt
```

Télécharger et décompresser le dataset Kaggle, puis placer `cardio_train.csv`
dans `TP2/data/` sur le cluster. Le CSV et les fichiers TensorBoard sont exclus de Git.

```bash
python dataset.py
```

`dataset.py` construit les ensembles 80/10/10. Le scaler est ajusté sur le train
uniquement pour corriger la fuite de données signalée dans l’énoncé.
La dimension d’entrée du MLP est lue depuis les données encodées (16 variables).

## Expériences réalisées

```bash
python train.py --l1 0.0001 --l2 0.001
python train.py --l1 0.1 --l2 0
python train.py --compare --epochs 30 --lr 0.001 --l1 0 --l2 0
tensorboard --logdir=runs
```

`train.py` conserve les historiques CSV dans `results/`, les événements TensorBoard
dans `runs/` et les meilleurs checkpoints selon la BCE de validation dans `checkpoints/`.

## Évaluation finale

```bash
python evaluate.py
```

Par défaut, cette commande charge le checkpoint RMSprop de l’expérience réalisée
sur le cluster, retenu à l’époque 11. Pour une nouvelle exécution de l’entraînement,
indiquer son checkpoint explicitement :

```bash
python evaluate.py --checkpoint checkpoints/NOM_EXPERIENCE/best.pt
```

Le CSV doit rester identique : le prétraitement et les partitions sont reconstruits
avec la graine sauvegardée. Les données, checkpoints et événements TensorBoard ne
sont pas inclus dans Git. Ils sont conservés sur le cluster ; une nouvelle copie du
dépôt nécessite de les récupérer ou de relancer les expériences.

Le compte rendu final et les captures sont dans `rapport.md` et `images/`.
