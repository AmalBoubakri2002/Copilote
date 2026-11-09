# Identification des utilisateurs du logiciel Copilote à partir de leurs traces d'utilisation

| | |
|---|---|
| **Enseignement** | MOD 7.2 – Introduction à la science des données, Centrale Lyon |
| **Encadrants** | Julien Velcin, Antoine Becquet |
| **Auteur** | Amal Boubakri, Mohamed Montassar Ben Youssef, Ichrak Ben Yahia |
| **Compétition** | Kaggle « Qui utilise mon appli ? v2026/2027 – Groupe 1 » |

## 1. Présentation

Ce dépôt contient la chaîne de traitement complète d'un problème de classification multi-classes : identifier, parmi 247 utilisateurs,
l'auteur d'une session du logiciel Copilote (éditeur Infologic) à partir de la seule trace de ses actions (actions, écrans, configurations, navigateur).
La métrique du défi est le **F1 macro** (moyenne des F1 de chaque utilisateur).

La solution retenue représente chaque session par des n-grammes pondérés TF-IDF, puis combine trois modèles complémentaires
(SVM linéaire, Extra Trees, KNN cosinus). Elle atteint un **F1 macro de 0,9762 en validation croisée** (accuracy 0,978).
Le meilleur score public obtenu sur le leaderboard du groupe est de **0,977**.

## 2. Installation et exécution


```bash
pip install -r requirements.txt

python src/etape1_decouverte.py       
python src/etape2_premier_modele.py   
python src/etape3_optimisation.py    
python src/main.py                   
```

Les étapes sont à exécuter dans cet ordre : l'étape 3 réutilise les scores enregistrés par l'étape 2.
`main.py` est indépendant des étapes précédentes et suffit pour produire le fichier de soumission.

## 3. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── data/                         
├── outputs/                       fichiers générés par les scripts
└── src/
    ├── data.py                    lecture et décomposition des traces
    ├── features.py                variables de fréquences relatives
    ├── modele.py                  représentation TF-IDF et modèle final
    ├── etape1_decouverte.py       Étape 1
    ├── etape2_premier_modele.py   Étape 2
    ├── etape3_optimisation.py     Étape 3
    └── main.py                    soumission Kaggle
```

| Fichier | Contenu | Fichiers produits |
|---|---|---|
| `data.py` | Lecture des traces ; décomposition de chaque action en cinq champs (base, écran, configuration, chaîne, modification) ; chemins et chargeurs communs. | — |
| `features.py` | Fréquences relatives des actions, écrans, configurations, chaînes, bigrammes et indicateurs de début de session ; vocabulaire appris sur le train. | — |
| `modele.py` | Blocs TF-IDF de n-grammes, KNN cosinus, Extra Trees et classe `Classifieur` (ensemble final). | — |
| `etape1_decouverte.py` | **Étape 1** : découverte des données, nettoyage, mise en forme tabulaire, premières explorations et visualisations. | `outputs/tableau_sessions.csv.gz`, `outputs/figures/etape1_exploration.png` |
| `etape2_premier_modele.py` | **Étape 2** : explorations avancées (jetons propres à un utilisateur, profils proches), référence naïve, premier modèle (Extra Trees), analyse des erreurs. | `outputs/figures/etape2_erreurs.png`, `outputs/cache/et_oof.npy` |
| `etape3_optimisation.py` | **Étape 3** : optimisation en trois parties (A : représentation ; B : navigateur et pondération des classes ; C : ensemble et correction de fréquence), puis variables influentes. | `outputs/figures/etape3_progression.png`, `outputs/cache/pli*.npz` |
| `main.py` | Entraînement du modèle final sur l'ensemble du train et écriture de la soumission. | `outputs/submission.csv` |

## 4. Méthodologie

**Données.** Le jeu d'entraînement compte 3 278 sessions (une session vide est supprimée) réparties sur 247 utilisateurs, de 4 à 75 sessions chacun (médiane : 12).
Le test compte 324 sessions. La comparaison train/test (longueur des sessions, navigateurs, vocabulaire) ne révèle pas de décalage de distribution.

**Représentation.** Chaque session est traitée comme un document. Six familles de jetons fournissent chacune un bloc TF-IDF
(pondération sous-linéaire ; jetons présents dans au moins deux sessions) :
actions (1 à 3-grammes), écrans, configurations, couples action/écran, couples configuration/chaîne et action complète à cinq champs (1 à 2-grammes chacun).
Le navigateur est ajouté en codage one-hot, pondéré par 3 dans le SVM.
Vocabulaires et pondérations IDF sont appris uniquement sur les données d'apprentissage du pli considéré, ce qui exclut toute fuite d'information.

**Modèle final.** Les scores des trois modèles sont normalisés ligne par ligne, puis combinés :

```
score = 0,8·z(SVM) + 0,1·z(Extra Trees) + 0,1·z(KNN cosinus) − 0,5·log(fréquence de l'utilisateur dans le train)
```

| Composante | Paramètres |
|---|---|
| SVM linéaire | `LinearSVC`, C = 1, classes équilibrées, TF-IDF et navigateur |
| Extra Trees | 500 arbres, classes équilibrées, `max_features=100`, `min_samples_leaf=3`, fréquences relatives |
| KNN cosinus | moyenne des 3 meilleures similarités cosinus de chaque utilisateur |

Le terme en logarithme de la fréquence atténue le biais du modèle en faveur des utilisateurs les plus représentés.

**Protocole de validation.** Validation croisée stratifiée en 4 plis (le plus petit utilisateur ne compte que 4 sessions), graine 0.
Toutes les expériences sont comparées sur exactement les mêmes plis, en ne modifiant qu'un élément à la fois.
Les sessions d'un même utilisateur sont réparties entre apprentissage et validation, comme entre train et test dans le défi.

## 5. Résultats

Résultats produits par `etape3_optimisation.py` (validation croisée stratifiée à 4 plis, graine 0) :

| Modèle | F1 macro | F1 sur tests simulés* |
|---|---|---|
| Étape 2 : Extra Trees (fréquences relatives) | 0,9515 | 0,9410 |
| A1 – n-grammes d'actions + SVM | 0,9194 | 0,9091 |
| A2 – + écrans et configurations | 0,9560 | 0,9452 |
| A3 – + couples action/écran et configuration/chaîne | 0,9675 | 0,9594 |
| A4 – + action complète | 0,9701 | 0,9619 |
| B1 – navigateur pondéré ×3 | 0,9709 | 0,9630 |
| B2 – classes équilibrées (SVM final) | 0,9724 | 0,9648 |
| C1 – SVM + Extra Trees (0,8 / 0,2) | 0,9757 | 0,9697 |
| C2 – + KNN cosinus (0,8 / 0,1 / 0,1) | 0,9750 | 0,9689 |
| **C3 – + correction de fréquence (β = 0,5) : modèle final** | **0,9762** | **0,9704** |

\* Moyenne sur 300 tirages de 324 sessions (découpage stratifié proportionnel), F1 macro calculé sur l'union des classes vraies et prédites.
Il s'agit d'une estimation de ce que donnerait un test de la taille du test réel, et non d'un score du leaderboard.

**Variables les plus influentes.** Pour le SVM final, la valeur absolue moyenne des coefficients (sur les 247 utilisateurs) se répartit ainsi :

| Famille de variables | Part de l'importance |
|---|---|
| Actions complètes (5 champs) | 28,3 % |
| Actions de base | 26,2 % |
| Couples action/écran | 21,1 % |
| Écrans | 10,3 % |
| Couples configuration/chaîne | 7,6 % |
| Configurations | 5,5 % |
| Navigateur | 1,0 % |

Parmi les jetons individuels dominent des configurations d'écran (`INSTALLATEUR`, `STD`, `DEF_07/23`, `ACCUEIL_INST`, `DEF_03/24`).
Le jeton `None|None`, qui correspond à l'absence de configuration et de chaîne, figure en tête du classement sans avoir de signification métier propre ; il doit être interprété avec prudence.

## 6. Analyse et limites

- **Apport de chaque étape.** Le gain principal provient du passage aux n-grammes TF-IDF avec un SVM (de 0,9515 à 0,9724), puis de l'ajout de l'Extra Trees (0,9757).
- **Composantes secondaires.** Le KNN et la correction de fréquence modifient le F1 macro de moins de 0,001 sur cette graine, ce qui est du même ordre que la variabilité de la mesure.
  Les poids (0,8 / 0,1 / 0,1) et le coefficient β = 0,5 ont été choisis lors d'une étude complémentaire sur plusieurs graines et sur des tests simulés.
- **Sessions courtes.** Les erreurs se concentrent sur les sessions courtes : avec l'Extra Trees de l'étape 2, le taux d'erreur est de 10,9 % sur le quart de sessions le plus court, contre 1,3 % sur le plus long (voir `etape2_erreurs.png`).
- **Biais d'optimisme.** Les configurations sont comparées sur les mêmes plis et les poids de l'ensemble ont été choisis à partir de scores de validation croisée ; le F1 macro final est donc légèrement surestimé.
- **Pistes.** Un traitement spécifique des sessions courtes, ou des modèles de séquence, constituent les principales perspectives d'amélioration.
