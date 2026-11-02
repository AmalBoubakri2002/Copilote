"""Partie 2 : exploration avancée + premiers modèles de classification.

Lancer depuis la racine :  python src/model.py  
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs")
FIG = OUT / "figures"
SEED = 0
lignes = []


def log(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    lignes.append(t)


d = joblib.load(OUT / "features_v2.joblib")
X, y, Xte = d["X_train"], d["y"], d["X_test"]
par_util = pd.Series(y).value_counts()

# ------------------------------------------------ 1. Exploration avancée

log("== 1. Exploration avancée ==")
# Les utilisateurs sont-ils plus homogènes entre eux qu'avec les autres ? (similarité cosinus)
A = X.filter(like="act__").to_numpy()
A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
S = A @ A.T
mm = y[:, None] == y[None, :]
np.fill_diagonal(mm, False)
off = ~(y[:, None] == y[None, :])
log(f"Similarité cosinus (profils d'actions) : même utilisateur {S[mm].mean():.3f} | "
    f"utilisateurs différents {S[off].mean():.3f}")
# Variables les plus stables : part de variance expliquée par l'utilisateur (inter / totale)
g = X.assign(u=y).groupby("u")
inter = ((g.transform("mean") - X.mean()) ** 2).sum()
tot = ((X - X.mean()) ** 2).sum()
eta2 = (inter / tot.replace(0, np.nan)).fillna(0).sort_values(ascending=False)
log("Top 8 variables par eta² (part de variance due à l'utilisateur) :")
for k, v in eta2.head(8).items():
    log(f"   {v:.3f}  {k}")
plt.figure(figsize=(6, 4))
plt.hist(eta2, bins=40, color="#3b6ea5")
plt.xlabel("eta² (variance expliquée par l'utilisateur)")
plt.ylabel("Nb variables")
plt.title("Pouvoir discriminant des variables")
plt.tight_layout()
plt.savefig(FIG / "08_eta2.png", dpi=130)
plt.close()

# ------------------------------------------------ 2. Protocole d'évaluation
# 4 plis stratifiés (le plus petit utilisateur a 4 sessions), score = F1 macro
# (moyenne des F1 par utilisateur, comme la métrique Kaggle "F1 moyen").
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
modeles = {
    "Dummy (classe majoritaire)": DummyClassifier(strategy="most_frequent"),
    "Régression logistique": make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=100, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, class_weight="balanced",
                                        max_features=100, n_jobs=-1, random_state=SEED),
}
log("\n== 2. Modèles (CV stratifiée 4 plis) ==")
preds, proba = {}, {}
for nom, m in modeles.items():
    p = cross_val_predict(m, X, y, cv=cv)
    preds[nom] = p
    log(f"{nom:28s} F1 macro {f1_score(y, p, average='macro'):.3f} | accuracy {accuracy_score(y, p):.3f}")

# ------------------------------------------------ 3. Analyse du meilleur modèle
best = max(preds, key=lambda k: f1_score(y, preds[k], average="macro"))
p = preds[best]
log(f"\n== 3. Analyse : {best} ==")
f1u = pd.Series(f1_score(y, p, average=None, labels=par_util.index), index=par_util.index)
rare = par_util[par_util <= 5].index
log(f"F1 moyen utilisateurs rares (<=5 sessions) : {f1u[rare].mean():.3f} | autres : {f1u.drop(rare).mean():.3f}")
err = pd.Series(list(zip(y[p != y], p[p != y]))).value_counts().head(6)
log("Confusions principales (vrai -> prédit) :", {f"{a}->{b}": int(n) for (a, b), n in err.items()})
plt.figure(figsize=(6, 4))
plt.scatter(par_util.values, f1u.values, s=14, color="#3b6ea5")
plt.xlabel("Nb de sessions d'entraînement de l'utilisateur")
plt.ylabel("F1 de l'utilisateur")
plt.title(f"{best} : F1 par utilisateur")
plt.tight_layout()
plt.savefig(FIG / "09_f1_par_utilisateur.png", dpi=130)
plt.close()

final = modeles[best].fit(X, y)
if hasattr(final, "feature_importances_"):
    imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("Top 10 variables influentes :", list(imp.head(10).index))
    imp.head(20)[::-1].plot.barh(figsize=(8, 6), color="#3b6ea5")
    plt.title(f"{best} : importance des variables")
    plt.tight_layout()
    plt.savefig(FIG / "10_importance_modele.png", dpi=130)
    plt.close()
    fam = imp.groupby(imp.index.str.split("__").str[0]).sum().sort_values(ascending=False)
    log("Importance cumulée par famille :", {k: round(float(v), 3) for k, v in fam.items()})

# ------------------------------------------------ 4. Prédictions test
sub = pd.DataFrame({"RowId": np.arange(1, len(Xte) + 1), "prediction": final.predict(Xte)})
sub.to_csv(OUT / "submission_part2.csv", index=False)
log(f"\nsubmission_part2.csv : {len(sub)} lignes, {sub['prediction'].nunique()} utilisateurs distincts prédits")
(OUT / "model_summary.txt").write_text("\n".join(lignes), encoding="utf-8")
