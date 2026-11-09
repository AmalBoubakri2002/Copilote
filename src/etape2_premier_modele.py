"""ÉTAPE 2 : explorations plus avancées des données et développement d'un premier modèle de classification.
Premier modèle : Extra Trees sur les fréquences relatives (features.py), validation croisée stratifiée
en 4 plis. Les scores hors-échantillon sont enregistrés dans outputs/cache/ pour l'étape 3.
Produit : outputs/figures/etape2_erreurs.png et outputs/cache/et_oof.npy
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.metrics import accuracy_score, f1_score      # noqa: E402
from sklearn.model_selection import StratifiedKFold       # noqa: E402

from data import SORTIES, charger_train                    # noqa: E402
from features import Constructeur, Debut                   # noqa: E402
from modele import creer_extra_trees                       # noqa: E402

GRAINE, N_PLIS = 0, 4


def titre(texte):
    print(f"\n{'=' * 90}\n{texte}\n{'=' * 90}")


train = charger_train()
classes = np.array(sorted(train["util"].unique()))
y = np.searchsorted(classes, train["util"].to_numpy())
K = len(classes)
taille = np.bincount(y, minlength=K)
rares = taille <= 5            # utilisateurs ayant 5 sessions ou moins

# ------------------------------------------------------------------ A. exploration avancée
titre("A. Quels éléments des traces sont propres à un utilisateur ?")
print("Pour chaque famille de jetons : part des jetons utilisés par un seul utilisateur (indice très discriminant).")
familles = {"action de base": 0, "écran": 1, "configuration": 2, "chaîne": 3}
lignes = []
for nom, i in familles.items():
    utilisateurs_par_jeton = {}
    for acts, u in zip(train["actions"], train["util"]):
        for a in acts:
            if a[i]:
                utilisateurs_par_jeton.setdefault(a[i], set()).add(u)
    nb = np.array([len(v) for v in utilisateurs_par_jeton.values()])
    lignes.append({"famille": nom, "jetons distincts": len(nb), "% utilisés par 1 seul utilisateur": 100 * (nb == 1).mean(),
                   "% par au plus 3 utilisateurs": 100 * (nb <= 3).mean()})
print(pd.DataFrame(lignes).set_index("famille").round(1).to_string())
print("-> beaucoup d'écrans, de configurations et de chaînes sont propres à quelques utilisateurs : pondérer les jetons rares "
      "(TF-IDF) sera utile à l'étape 3.")

print("\nQuels utilisateurs ont des comportements proches ? (cosinus entre profils moyens, actions et écrans)")
F = Constructeur().fit(train).transform(train)             # exploration seulement : vocabulaire appris sur tout le train
colonnes = [c for c in F.columns if c.startswith(("act__", "ecr__"))]
profils = np.array([F.loc[y == k, colonnes].mean().to_numpy() for k in range(K)])
profils /= np.linalg.norm(profils, axis=1, keepdims=True) + 1e-12
cos = profils @ profils.T
np.fill_diagonal(cos, -1)
paires = sorted({tuple(sorted((i, j))) for i, j in zip(*np.unravel_index(np.argsort(-cos, axis=None)[:12], cos.shape))},
                key=lambda p: -cos[p])[:5]
for i, j in paires:
    print(f"   {classes[i]} / {classes[j]} : cosinus {cos[i, j]:.3f}  (sessions : {taille[i]} et {taille[j]})")
print(f"   cosinus médian entre deux utilisateurs : {np.median(cos[cos > -1]):.3f}")

# ------------------------------------------------------------------ B. référence naïve
titre("B. Référence naïve : toujours prédire l'utilisateur le plus fréquent")
naif = np.full(len(y), np.bincount(y).argmax())
print(f"F1 macro {f1_score(y, naif, average='macro'):.4f} | accuracy {accuracy_score(y, naif):.4f}")

# ------------------------------------------------------------------ C. premier modèle : Extra Trees
titre("C. Premier modèle : Extra Trees sur les fréquences relatives (validation croisée 4 plis)")
probas = np.zeros((len(y), K), dtype=np.float32)
acc_plis = []
for apprentissage, validation in StratifiedKFold(N_PLIS, shuffle=True, random_state=GRAINE).split(train, y):
    d_app, d_val = train.iloc[apprentissage], train.iloc[validation]
    constructeur, debut = Constructeur().fit(d_app), Debut().fit(d_app)          # vocabulaire appris sur le pli seulement
    X_app = pd.concat([constructeur.transform(d_app), debut.transform(d_app)], axis=1)
    X_val = pd.concat([constructeur.transform(d_val), debut.transform(d_val)], axis=1)
    et = creer_extra_trees(graine=GRAINE).fit(X_app, y[apprentissage])
    probas[validation] = et.predict_proba(X_val)
    acc_plis.append(accuracy_score(y[validation], probas[validation].argmax(1)))
    print(f"pli terminé : accuracy {acc_plis[-1]:.4f}", flush=True)
pred = probas.argmax(1)
f1_par_classe = f1_score(y, pred, average=None, labels=np.arange(K))
print(f"\nF1 macro {f1_score(y, pred, average='macro'):.4f} | accuracy {accuracy_score(y, pred):.4f} "
      f"(écart-type entre plis {np.std(acc_plis):.4f})")
print(f"F1 des utilisateurs rares (<= 5 sessions) : {f1_par_classe[rares].mean():.4f} | des autres : {f1_par_classe[~rares].mean():.4f}")
(SORTIES / "cache").mkdir(parents=True, exist_ok=True)
np.save(SORTIES / "cache" / "et_oof.npy", probas)

# ------------------------------------------------------------------ D. analyse des erreurs et importance
titre("D. Où le modèle se trompe-t-il ? Quelles variables utilise-t-il ?")
erreur = pred != y
longueur = train["actions"].apply(len).to_numpy()
quartile = pd.qcut(longueur, 4, labels=["Q1 (courtes)", "Q2", "Q3", "Q4 (longues)"])
par_longueur = pd.Series(erreur).groupby(quartile, observed=True).mean() * 100
par_nav = pd.Series(erreur).groupby(train["navigateur"].to_numpy()).mean() * 100
par_nav = par_nav[train["navigateur"].value_counts().index[:3]]
print("taux d'erreur par longueur de session (%) :")
print(par_longueur.round(1).to_string())
print("taux d'erreur par navigateur (%) :")
print(par_nav.round(1).to_string())

constructeur, debut = Constructeur().fit(train), Debut().fit(train)
X = pd.concat([constructeur.transform(train), debut.transform(train)], axis=1)
et = creer_extra_trees(n_arbres=300, graine=GRAINE).fit(X, y)
top = pd.Series(et.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\n10 variables les plus importantes de l'Extra Trees (modèle entraîné sur tout le train) :")
for nom, v in top.items():
    print(f"   {v:.4f}  {nom[:90]}")

(SORTIES / "figures").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].bar(range(4), par_longueur.values, color="#4c78a8")
axes[0].set(xticks=range(4), xticklabels=["Q1\ncourtes", "Q2", "Q3", "Q4\nlongues"], ylabel="taux d'erreur (%)",
            title="Erreurs selon la longueur de la session")
axes[1].bar(range(3), par_nav.values, color="#4c78a8")
axes[1].set(xticks=range(3), xticklabels=[n.replace("Google ", "").replace("Microsoft ", "") for n in par_nav.index],
            title="Erreurs selon le navigateur", ylabel="taux d'erreur (%)")
plt.tight_layout()
plt.savefig(SORTIES / "figures" / "etape2_erreurs.png", dpi=150)
print("\nfigure enregistrée : outputs/figures/etape2_erreurs.png ; scores enregistrés : outputs/cache/et_oof.npy")
