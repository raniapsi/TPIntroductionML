# TP2 — Régularisation, optimisation et métriques

Dans ce TP, je travaille sur la classification de données cardiovasculaires avec PyTorch. L’objectif est de préparer les données, puis d’étudier les régularisations L1/L2, de comparer plusieurs optimiseurs et d’évaluer les prédictions du modèle.

> Rapport en cours : la préparation des données a été vérifiée sur le cluster. Les expériences d’entraînement restent à réaliser.

## 1. Dataset personnalisé

### Chargement et préparation des données

J’ai placé le fichier `cardio_train.csv` dans `TP2/data/`. Le script `dataset.py` lit le CSV avec le séparateur `;`, applique la suppression des doublons prévue dans le code et retire la colonne `id`. La colonne `cardio` sert de cible : 0 pour l’absence de maladie cardiovasculaire et 1 pour sa présence.

Les variables catégorielles `gender`, `cholesterol` et `gluc` sont transformées par encodage one-hot : une colonne indicatrice est créée pour chaque catégorie. Après cette préparation, chaque patient est représenté par 16 variables.

### Interface PyTorch

`__len__` renvoie le nombre d’exemples. `__getitem__` renvoie les variables
et le label de l’exemple demandé. Le DataLoader rassemble ces exemples en lots.
On mélange les données d’entraînement ; validation et test ne nécessitent pas de mélange.

### Fuite de données (data leakage)

Ajuster StandardScaler sur toutes les données utilise la moyenne et l’écart-type
des ensembles de validation et de test pour préparer l’entraînement.
Ces ensembles doivent rester indépendants de cet ajustement.
Dans notre code, on effectue donc le découpage avant d’ajuster le scaler sur le train,
puis on applique la même transformation aux trois ensembles.

### Données trop volumineuses pour la RAM

On utiliserait `torch.utils.data.IterableDataset` avec une lecture progressive
(par exemple des blocs CSV via `pandas.read_csv(..., chunksize=...)`).
Cette classe ne suffit pas seule : la lecture doit aussi éviter de tout charger en mémoire.

### Vérification sur le cluster

J’ai exécuté le premier script sur le nœud de calcul `starfighter-slurm-node-01-1`, dans le job Slurm 4628, avec l’environnement `deeplearning` :

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
cd ~/TPIntroductionML/TP2
python dataset.py
```

![Exécution de dataset.py sur le cluster : tailles des ensembles et dimensions du premier batch](images/dataset-cluster.png)

Le script affiche les résultats suivants :

```text
Tailles train / validation / test : [56000, 7000, 7000]
Shape features: torch.Size([64, 16]) Shape labels: torch.Size([64, 1])
Nombre de variables après encodage : 16
```

Les 70 000 exemples sont répartis en 56 000 exemples pour l’entraînement (80 %), 7 000 pour la validation (10 %) et 7 000 pour le test (10 %). Le découpage utilise une graine fixée à 42 pour être reproductible.

La forme `[64, 16]` correspond à un batch de 64 patients possédant chacun 16 variables. La forme `[64, 1]` signifie que chaque patient est associé à un seul label binaire. Ces résultats confirment que le dataset et les DataLoaders peuvent fournir les lots attendus pour l’entraînement.

Pour la suite, la dimension d’entrée du MLP sera déterminée à partir des données : elle vaut ici 16. Il ne faut donc pas reprendre la valeur fixe de 12 figurant dans l’un des exemples de l’énoncé.

## 2. MLP et régularisations L1/L2

À faire : implémentation, expérience avec L1 forte, observations et réponses théoriques.

## 3. Optimiseurs et TensorBoard

À faire : comparer SGD, Momentum, RMSprop et Adam ; insérer la capture et commenter les courbes.

## 4. Métriques

À faire : choisir le modèle sur la validation, évaluer sur le test et interpréter les métriques.
