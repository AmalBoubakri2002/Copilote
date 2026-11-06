# Identification des utilisateurs de Copilote (MOD 7.2)

Objectif : retrouver l'utilisateur d'un logiciel (247 utilisateurs) à partir d'une session de traces d'utilisation
(séquence d'actions, écrans, configurations, navigateur). Métrique du challenge : F1 macro.

## Organisation : les trois étapes demandées + le modèle final

| Fichier | Rôle |
|---|---|
| `src/data.py` | lecture et décomposition des traces brutes (action = base, écran, configuration, chaîne, modification) |
| `src/features.py` | variables « fréquences relatives » (actions, écrans, configurations, chaînes, bigrammes, début de session) |
| `src/modele.py` | représentation TF-IDF de n-grammes et modèle final (ensemble de trois modèles) |
| **`src/etape1_decouverte.py`** | **Étape 1** : découverte, nettoyage, mise en forme tabulaire, premières explorations et visualisations |
| **`src/etape2_premier_modele.py`** | **Étape 2** : explorations avancées et premier modèle (Extra Trees) |
| **`src/etape3_optimisation.py`** | **Étape 3** : optimisation (représentation, navigateur, classes, ensemble, correction de fréquence) et variables influentes |
| `src/main.py` | modèle final : entraîne sur tout le train et écrit la soumission Kaggle |

## Utilisation

Python 3.12 testé ici ; le code n'utilise que pandas, numpy, scikit-learn, scipy et matplotlib.

    pip install -r requirements.txt
    # placer data/train.csv.GZ, data/test.csv.GZ, data/sample_submission.csv dans data/
    python3 src/etape1_decouverte.py          # ~1 min   -> outputs/tableau_sessions.csv.gz, outputs/figures/etape1_exploration.png
    python3 src/etape2_premier_modele.py      # ~4 min   -> outputs/figures/etape2_erreurs.png (Extra Trees, F1 macro 0,9515)
    python3 src/etape3_optimisation.py        # ~8 min   -> outputs/figures/etape3_progression.png + variables influentes (reprend si interrompu)
    python3 src/main.py                       # ~2 min   -> outputs/submission.csv (à déposer sur Kaggle)

L'étape 3 utilise le cache de l'étape 2 : lancer les étapes dans l'ordre.

## Résultats mesurés (validation croisée stratifiée 4 plis, graine 0, mêmes plis partout)

| Modèle | F1 macro | F1 test simulé* |
|---|---|---|
| Étape 2 : Extra Trees (fréquences relatives) | 0,9515 | 0,9410 |
| A1 n-grammes d'actions + SVM | 0,9194 | 0,9091 |
| A2 + écrans et configurations | 0,9560 | 0,9452 |
| A3 + couples action\|écran et configuration\|chaîne | 0,9675 | 0,9594 |
| A4 + action complète | 0,9701 | 0,9619 |
| B1 navigateur pondéré x3 | 0,9709 | 0,9630 |
| B2 classes équilibrées (SVM final) | 0,9724 | 0,9648 |
| C1 SVM + Extra Trees (0,8 / 0,2) | 0,9757 | 0,9697 |
| C2 + KNN cosinus (0,8 / 0,1 / 0,1) | 0,9750 | 0,9689 |
| C3 + correction du biais de fréquence = **modèle final** | **0,9762** | **0,9704** |

*moyenne sur 300 tirages de 324 sessions (découpage stratifié proportionnel), F1 macro sur l'union des classes.

À retenir honnêtement : l'essentiel du gain vient de la représentation TF-IDF et du SVM (de 0,9515 à 0,9724) puis de
l'ajout de l'Extra Trees (0,9757). Le KNN et la correction de fréquence (C2, C3) changent le F1 de moins de 0,001 sur
cette graine : leur apport est dans le bruit de la mesure. Les poids (0,8 / 0,1 / 0,1) et beta = 0,5 ont été choisis lors d'une étude
plus large (3 graines, tests simulés), pas sur ce seul tableau.
Meilleur score public Kaggle du groupe : 0,977 ; `outputs/submission.csv` est identique, ligne pour ligne, à la
soumission V3 (`submission_optimisation_v3.csv`) ; à confirmer en la soumettant.
