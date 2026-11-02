"""Étapes 3-4 : qualité des données, nettoyage, tableau final, visualisations.

Lancer depuis la racine du projet :  python src/explore.py
Sorties : outputs/figures/*.png, outputs/exploration_summary.txt, outputs/features_v2.joblib
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data import charger
from features import Constructeur

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
lignes = []


def log(*args):
    txt = " ".join(str(a) for a in args)
    print(txt)
    lignes.append(txt)


def sauver(nom):
    plt.tight_layout()
    plt.savefig(FIG / nom, dpi=130)
    plt.close()


# ---------------------------------------------------------------- 1. Chargement
tr = charger("data/train.csv.GZ", avec_util=True)
te = charger("data/test.csv.GZ", avec_util=False)
log("== 1. Découverte ==")
log(f"Sessions train : {len(tr)} | test : {len(te)}")
log(f"Utilisateurs (train) : {tr['util'].nunique()}")
par_util = tr["util"].value_counts()
log(f"Sessions par utilisateur : min {par_util.min()}, médiane {par_util.median():.0f}, max {par_util.max()}")
log(f"Utilisateurs avec <= 5 sessions : {(par_util <= 5).sum()}")
log("Navigateurs train :", tr["navigateur"].value_counts().to_dict())
log("Navigateurs test  :", te["navigateur"].value_counts().to_dict())
nb_nav = tr.groupby("util")["navigateur"].nunique()
log(f"Utilisateurs utilisant > 1 navigateur : {(nb_nav > 1).sum()}")

# ---------------------------------------------------------------- 2. Qualité / nettoyage
log("\n== 2. Qualité et nettoyage ==")
vides_tr = tr["actions"].apply(len) == 0
vides_te = te["actions"].apply(len) == 0
log(f"Sessions vides : train {vides_tr.sum()}, test {vides_te.sum()}")
tr_clean = tr[~vides_tr].copy()          # on ne supprime JAMAIS une ligne du test (324 attendues)
log(f"Train après suppression des sessions vides : {len(tr_clean)}")
cle = tr_clean["actions"].apply(lambda a: tuple(a))
log(f"Sessions dupliquées dans le train : {cle.duplicated().sum()}")
inconnus = set(te["navigateur"]) - set(tr["navigateur"])
log(f"Navigateurs du test absents du train : {inconnus or 'aucun'}")

# ---------------------------------------------------------------- 3. Tableau
cons = Constructeur().fit(tr_clean)
X_train = cons.transform(tr_clean)
X_test = cons.transform(te)
y = tr_clean["util"].to_numpy()
const = X_train.columns[X_train.nunique() <= 1]
X_train, X_test = X_train.drop(columns=const), X_test.drop(columns=const)
log("\n== 3. Tableau ==")
log(f"Variables constantes supprimées : {len(const)}")
log(f"X_train {X_train.shape} | X_test {X_test.shape} | NaN : {int(X_train.isna().sum().sum())}")
for fam in ["nav", "act", "ecr", "conf", "chn", "bi"]:
    log(f"  famille {fam:5s}: {sum(c.startswith(fam + '__') for c in X_train.columns)} variables")
joblib.dump({"X_train": X_train, "y": y, "X_test": X_test}, OUT / "features_v2.joblib")

# ---------------------------------------------------------------- 4. Visualisations
sns_style = {"figure.figsize": (7, 4)}
plt.rcParams.update(sns_style)

plt.hist(par_util, bins=40, color="#3b6ea5")
plt.xlabel("Nombre de sessions par utilisateur"); plt.ylabel("Nb utilisateurs")
plt.title("Classes déséquilibrées : sessions par utilisateur"); sauver("01_sessions_par_utilisateur.png")

n = X_train["n_actions"]
plt.hist(np.log10(n), bins=40, color="#3b6ea5")
plt.xlabel("log10(nombre d'actions)"); plt.ylabel("Nb sessions")
plt.title("Longueur des sessions (échelle log)"); sauver("02_longueur_sessions.png")

nav = pd.concat([tr["navigateur"].value_counts(normalize=True).rename("train"),
                 te["navigateur"].value_counts(normalize=True).rename("test")], axis=1)
nav.plot.bar(rot=15); plt.ylabel("Part des sessions")
plt.title("Navigateurs : train vs test"); sauver("03_navigateurs.png")

act = X_train.filter(like="act__").mean().sort_values(ascending=False).head(15)
act.index = act.index.str.replace("act__", "")
act[::-1].plot.barh(color="#3b6ea5"); plt.xlabel("Fréquence relative moyenne")
plt.title("Actions les plus fréquentes"); sauver("04_actions_frequentes.png")

top6 = par_util.index[:6]
prof = (X_train.filter(like="act__").assign(util=y).groupby("util").mean().loc[top6])
prof = prof[act.index.map(lambda a: "act__" + a)]
prof.columns = [c.replace("act__", "")[:25] for c in prof.columns]
plt.figure(figsize=(9, 4)); plt.imshow(prof.values, aspect="auto", cmap="viridis")
plt.xticks(range(prof.shape[1]), prof.columns, rotation=75, ha="right", fontsize=7)
plt.yticks(range(6), prof.index); plt.colorbar(label="fréquence")
plt.title("Profils d'actions de 6 utilisateurs"); sauver("05_profils_utilisateurs.png")

Xs = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=20, random_state=0).fit(Xs)
Z = pca.transform(Xs)
log("\n== 4. Exploration ==")
log(f"PCA : 2 composantes = {pca.explained_variance_ratio_[:2].sum():.1%} de variance, "
    f"20 composantes = {pca.explained_variance_ratio_.sum():.1%}")
for u in top6:
    m = y == u
    plt.scatter(Z[m, 0], Z[m, 1], s=10, label=u, alpha=.7)
plt.legend(title="util"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("ACP : 6 utilisateurs les plus actifs"); sauver("06_pca.png")

F, _ = f_classif(X_train.fillna(0), y)
imp = pd.Series(np.nan_to_num(F, posinf=0), index=X_train.columns).sort_values(ascending=False)
log("Top 10 variables (F ANOVA) :", list(imp.head(10).index))
log(f"Variables avec F < 1 (peu discriminantes) : {(imp < 1).sum()} / {len(imp)}")
imp.head(20)[::-1].plot.barh(color="#3b6ea5", figsize=(8, 6)); plt.xlabel("F de l'ANOVA")
plt.title("Variables les plus discriminantes"); sauver("07_importance_anova.png")

rare = par_util[par_util <= 5].index
log(f"Sessions d'utilisateurs rares (<=5 sessions) : {np.isin(y, rare).sum()}")
(OUT / "exploration_summary.txt").write_text("\n".join(lignes), encoding="utf-8")
