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

### Architecture et fonction de coût

Le script `train.py` définit un MLP avec 16 entrées, deux couches cachées de 128 neurones avec activation ReLU, puis une sortie avec activation sigmoïde. Cette sortie représente la probabilité de la classe 1. La fonction de perte utilisée est la binary cross-entropy (`BCELoss`).

L’objectif optimisé ajoute deux pénalités à cette perte :

$$J = \mathrm{BCE} + \lambda_1 \sum_p |p| + \lambda_2 \sum_p p^2.$$

Comme dans le code de l’énoncé, ces sommes portent sur tous les paramètres, biais compris. À chaque batch, `zero_grad()` efface les gradients précédents, `backward()` calcule les nouveaux gradients et `step()` met à jour les paramètres.

### Protocole de comparaison

Deux expériences sont préparées avec SGD, un taux d’apprentissage de 0,01, 10 époques et la même graine :

```bash
python train.py --l1 0.0001 --l2 0.001
python train.py --l1 0.1 --l2 0
```

Le script enregistre les résultats par époque dans `results/` au format CSV et dans `runs/` pour TensorBoard. Il distingue l’objectif avec pénalités de la BCE seule, afin de comparer les performances prédictives malgré des coefficients de régularisation différents. L’accuracy est calculée au seuil de 0,5. L’ensemble de test n’est pas utilisé à cette étape.

**Observations :** à compléter après les deux exécutions sur le cluster.

### Effet attendu d’une régularisation trop forte

Une pénalité trop forte peut empêcher le réseau d’apprendre les relations utiles : c’est le sous-apprentissage (underfitting). Il faudra vérifier cet effet dans les pertes et les accuracies mesurées, sans le confondre avec la hausse mécanique de l’objectif due à la pénalité.

### Régularisation L2 dans l’optimiseur

Avec `torch.optim.SGD`, l’argument `weight_decay` permet d’appliquer la régularisation L2. Pour reproduire une pénalité écrite sous la forme `l2_lambda * somme(p²)`, le coefficient équivalent est `weight_decay=2*l2_lambda`, car la dérivée de cette pénalité est `2*l2_lambda*p`. Il ne faut pas cumuler les deux mécanismes pour la même pénalité.

### Différence entre L1 et L2

La régularisation L1 favorise des paramètres nuls ou proches de zéro et peut produire une solution parcimonieuse. L2 réduit les grandes valeurs des paramètres de manière plus progressive, sans favoriser autant leur annulation. Avec les mises à jour SGD utilisées ici, L1 ne garantit pas des zéros exacts.

## 3. Optimiseurs et TensorBoard

À faire : comparer SGD, Momentum, RMSprop et Adam ; insérer la capture et commenter les courbes.

## 4. Métriques

À faire : choisir le modèle sur la validation, évaluer sur le test et interpréter les métriques.
