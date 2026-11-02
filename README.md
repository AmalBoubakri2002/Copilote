# Copilote – Qui utilise mon appli ? (MOD 7.2) – Partie 1

Placer `train.csv.GZ`, `test.csv.GZ`, `sample_submission.csv` dans `data/` (non versionnés), puis :

    pip install -r requirements.txt
    python src/explore.py

- `src/data.py`     : parsing des traces (action, écran, configuration, chaîne, modification, temps)
- `src/features.py` : construction des variables (fréquences relatives, bigrammes, variables globales)
- `src/explore.py`  : qualité, nettoyage, tableau final, figures dans `outputs/figures/`
