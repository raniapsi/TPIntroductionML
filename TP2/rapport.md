# TP2 — Régularisation, optimisation et métriques

> Rapport en cours. Aucune expérience d’entraînement n’a encore été exécutée.

## 1. Dataset personnalisé

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

À compléter après `python dataset.py` : tailles des ensembles et dimensions d’un batch.

## 2. MLP et régularisations L1/L2

À faire : implémentation, expérience avec L1 forte, observations et réponses théoriques.

## 3. Optimiseurs et TensorBoard

À faire : comparer SGD, Momentum, RMSprop et Adam ; insérer la capture et commenter les courbes.

## 4. Métriques

À faire : choisir le modèle sur la validation, évaluer sur le test et interpréter les métriques.
