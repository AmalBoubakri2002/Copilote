"""Étape 2 : construction des variables (une ligne = une session).

Familles de variables :
  - navigateur          : one-hot (4 navigateurs)
  - globales            : nb d'actions, durée, rythme, part en modification, ...
  - act__ / ecr__ / conf__ / chn__ : fréquences relatives des actions de base,
        écrans, configurations d'écran et chaînes (vocabulaire appris sur le TRAIN,
        seuil min_df sessions)
  - bi__                : fréquences relatives des bigrammes d'actions consécutives
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

NAVIGATEURS = ["Firefox", "Google Chrome", "Microsoft Edge", "Opera"]
MIN_DF = 5   # un jeton doit apparaître dans >= 5 sessions du train


def _jetons(df):
    """Pour chaque session, une liste de jetons par famille."""
    fam = {"act": [], "ecr": [], "conf": [], "chn": [], "bi": []}
    for acts in df["actions"]:
        base = [a[0] for a in acts]
        fam["act"].append(base)
        fam["ecr"].append([a[1] for a in acts if a[1]])
        fam["conf"].append([a[2] for a in acts if a[2]])
        fam["chn"].append([a[3] for a in acts if a[3]])
        fam["bi"].append([f"{x} > {y}" for x, y in zip(base, base[1:])])
    return fam


def _globales(df):
    n = df["actions"].apply(len)
    duree = df["temps"].apply(lambda t: max(t) if len(t) else 0)
    g = pd.DataFrame(index=df.index)
    g["n_actions"] = n
    g["log_n_actions"] = np.log1p(n)
    g["duree_s"] = duree
    g["actions_par_min"] = n / np.maximum(duree, 5) * 60
    g["part_modif"] = df["actions"].apply(lambda a: np.mean([x[4] for x in a]) if a else 0)
    g["n_ecrans_distincts"] = df["actions"].apply(lambda a: len({x[1] for x in a if x[1]}))
    g["n_chaines_distinctes"] = df["actions"].apply(lambda a: len({x[3] for x in a if x[3]}))
    g["part_actions_avec_ecran"] = df["actions"].apply(
        lambda a: np.mean([x[1] is not None for x in a]) if a else 0)
    return g


class Constructeur:
    """fit() apprend les vocabulaires sur le train ; transform() les applique."""

    def fit(self, df_train):
        fam = _jetons(df_train)
        self.vecs = {}
        for nom, docs in fam.items():
            v = CountVectorizer(analyzer=lambda x: x, min_df=MIN_DF)
            v.fit(docs)
            self.vecs[nom] = v
        return self

    def transform(self, df):
        n = df["actions"].apply(len).to_numpy().astype(float)
        n_safe = np.maximum(n, 1)[:, None]
        blocs = []
        nav = pd.DataFrame({f"nav__{k}": (df["navigateur"] == k).astype(int) for k in NAVIGATEURS},
                           index=df.index)
        blocs += [nav, _globales(df)]
        fam = _jetons(df)
        for nom, v in self.vecs.items():
            M = v.transform(fam[nom]).toarray() / n_safe      # fréquence relative
            cols = [f"{nom}__{t}" for t in v.get_feature_names_out()]
            blocs.append(pd.DataFrame(M, index=df.index, columns=cols))
        return pd.concat(blocs, axis=1)
