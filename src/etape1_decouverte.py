"""ÉTAPE 1 : découverte des données, nettoyage, mise en forme tabulaire, premières explorations et visualisations.

    python src/etape1_decouverte.py

Produit :
    outputs/tableau_sessions.csv.gz        une ligne = une session, variables globales + utilisateur + navigateur
    outputs/figures/etape1_exploration.png  quatre graphiques de découverte
"""
import gzip

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from data import DATA, SORTIES, charger_test, charger_train      # noqa: E402
from features import Constructeur, variables_globales            # noqa: E402


def titre(texte):
    print(f"\n{'=' * 90}\n{texte}\n{'=' * 90}")


# ------------------------------------------------------------------ 1. lecture et structure
titre("1. Lecture et structure des données")
with gzip.open(DATA / "train.csv.GZ", "rt", encoding="utf-8") as f:
    ligne_brute = f.readline()
print("Début de la première ligne brute :", ligne_brute[:230], "...")
train, test = charger_train(garder_vides=True), charger_test()
print("\nLa même session après décomposition (action = base, écran, configuration, chaîne, modification) :")
for action in train.loc[0, "actions"][:3]:
    print("   ", action)
print(f"\ntrain : {len(train)} sessions | test : {len(test)} sessions | utilisateurs du train : {train['util'].nunique()}")

# ------------------------------------------------------------------ 2. nettoyage
titre("2. Nettoyage et contrôles de qualité")
vides_train = (train["actions"].apply(len) == 0).sum()
vides_test = (test["actions"].apply(len) == 0).sum()
print(f"sessions vides : {vides_train} dans le train (supprimée), {vides_test} dans le test (jamais supprimée)")
train = train[train["actions"].apply(len) > 0].reset_index(drop=True)
print(f"navigateurs manquants : {train['navigateur'].isna().sum()} (train), {test['navigateur'].isna().sum()} (test)")
toutes = [a for acts in train["actions"] for a in acts]
for i, champ in enumerate(["écran", "configuration", "chaîne"], start=1):
    absent = np.mean([a[i] is None for a in toutes])
    print(f"actions sans {champ:13s}: {absent:6.1%}  (champ facultatif : ce n'est pas une valeur manquante à corriger)")
doublons = train.assign(seq=train["actions"].apply(tuple)).duplicated(subset=["util", "seq"]).sum()
print(f"sessions strictement identiques (même utilisateur, mêmes actions) : {doublons}")

# ------------------------------------------------------------------ 3. mise en forme tabulaire
titre("3. Mise en forme tabulaire : une ligne = une session")
globales = variables_globales(train)
tableau = pd.concat([train[["util", "navigateur"]], globales], axis=1)
SORTIES.mkdir(exist_ok=True)
tableau.to_csv(SORTIES / "tableau_sessions.csv.gz", index_label="session")
print(f"tableau des variables globales : {tableau.shape} -> outputs/tableau_sessions.csv.gz")
print(tableau.drop(columns=["util", "navigateur"]).describe().loc[["mean", "50%", "max"]].round(1).T.to_string())
matrice = Constructeur().fit(train).transform(train)
print(f"\nmatrice complète de variables (navigateur, globales, fréquences d'actions, écrans, configs, chaînes, bigrammes) : "
      f"{matrice.shape[0]} sessions x {matrice.shape[1]} variables")

# ------------------------------------------------------------------ 4. statistiques descriptives
titre("4. Premières explorations")
par_util = train["util"].value_counts()
print(f"sessions par utilisateur : min {par_util.min()}, médiane {par_util.median():.0f}, max {par_util.max()} ;"
      f" {(par_util <= 5).sum()} utilisateurs ont 5 sessions ou moins")
longueur = lambda d: d["actions"].apply(len)                                   # noqa: E731
duree = lambda d: d["temps"].apply(lambda t: max(t) if len(t) else 0)           # noqa: E731
comparaison = pd.DataFrame({
    "train": [longueur(train).median(), duree(train).median(), *train["navigateur"].value_counts(normalize=True).reindex(
        ["Firefox", "Google Chrome", "Microsoft Edge"]).mul(100)],
    "test": [longueur(test).median(), duree(test).median(), *test["navigateur"].value_counts(normalize=True).reindex(
        ["Firefox", "Google Chrome", "Microsoft Edge"]).mul(100)]},
    index=["longueur médiane (actions)", "durée médiane (s)", "% Firefox", "% Chrome", "% Edge"])
print("\nLe test ressemble-t-il au train ?")
print(comparaison.round(1).to_string())
top = pd.Series([a[0] for a in toutes]).value_counts(normalize=True).head(5).mul(100).round(1)
print("\nactions les plus fréquentes (% de toutes les actions) :")
print(top.to_string())

# ------------------------------------------------------------------ 5. visualisations
titre("5. Visualisations")
(SORTIES / "figures").mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

axes[0, 0].hist(par_util, bins=30, color="#4c78a8")
axes[0, 0].set(title="Sessions par utilisateur (train)", xlabel="nombre de sessions", ylabel="utilisateurs")

bins = np.logspace(0, 4, 40)
for nom, d, couleur in (("train", train, "#4c78a8"), ("test", test, "#f58518")):
    n = longueur(d)
    axes[0, 1].hist(n, bins=bins, weights=np.ones(len(n)) / len(n), alpha=0.6, label=nom, color=couleur)
axes[0, 1].set(xscale="log", title="Longueur des sessions", xlabel="nombre d'actions (échelle log)",
               ylabel="part des sessions")
axes[0, 1].legend()

noms_nav = ["Firefox", "Google Chrome", "Microsoft Edge", "Opera"]
x = np.arange(len(noms_nav))
for decalage, (nom, d, couleur) in zip((-0.2, 0.2), (("train", train, "#4c78a8"), ("test", test, "#f58518"))):
    part = d["navigateur"].value_counts(normalize=True).reindex(noms_nav).fillna(0) * 100
    axes[1, 0].bar(x + decalage, part, width=0.4, label=nom, color=couleur)
axes[1, 0].set(xticks=x, xticklabels=["Firefox", "Chrome", "Edge", "Opera"], title="Navigateurs", ylabel="% des sessions")
axes[1, 0].legend()

frequences = pd.Series([a[0] for a in toutes]).value_counts(normalize=True).head(12).mul(100)[::-1]
axes[1, 1].barh([n[:38] for n in frequences.index], frequences.values, color="#4c78a8")
axes[1, 1].set(title="12 actions les plus fréquentes", xlabel="% des actions")

plt.tight_layout()
plt.savefig(SORTIES / "figures" / "etape1_exploration.png", dpi=150)
print("figure enregistrée : outputs/figures/etape1_exploration.png")
