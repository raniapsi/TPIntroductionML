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
La dimension d’entrée du futur MLP devra être lue depuis les données encodées.

## Suite à faire ensemble

1. Vérifier les tailles et les dimensions affichées par le premier exercice.
2. Créer `train.py` : MLP et régularisations L1/L2.
3. Comparer les optimiseurs dans TensorBoard.
4. Évaluer le modèle choisi sur le test et compléter `rapport.md`.
