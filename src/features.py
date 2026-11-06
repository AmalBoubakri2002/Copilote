"""Étape 2 : variables « fréquences » (une ligne = une session), utilisées par l'Extra Trees.

Deux constructeurs, tous deux appris sur le train (vocabulaire) puis appliqués à n'importe quel jeu :

  Constructeur : navigateur (one-hot), variables globales (nombre d'actions, durée, rythme...),
                 fréquences relatives des actions, écrans, configurations, chaînes
                 et bigrammes d'actions consécutives.
  Debut        : indicateurs binaires sur la façon dont la session démarre
                 (premier écran, première action, trois premières actions...).

Un jeton n'est conservé que s'il apparaît dans au moins MIN_DF sessions du train.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

NAVIGATEURS = ["Firefox", "Google Chrome", "Microsoft Edge", "Opera"]
MIN_DF = 5


def identite(jetons):
    """Analyseur de CountVectorizer : les jetons sont déjà découpés, on les garde tels quels."""
    return jetons


# ------------------------------------------------------------------ Constructeur
def jetons_par_famille(df):
    """Pour chaque session, la liste de jetons de chaque famille."""
    familles = {"act": [], "ecr": [], "conf": [], "chn": [], "bi": []}
    for actions in df["actions"]:
        base = [a[0] for a in actions]
        familles["act"].append(base)
        familles["ecr"].append([a[1] for a in actions if a[1]])
        familles["conf"].append([a[2] for a in actions if a[2]])
        familles["chn"].append([a[3] for a in actions if a[3]])
        familles["bi"].append([f"{x} > {y}" for x, y in zip(base, base[1:])])
    return familles


def variables_globales(df):
    """Statistiques globales de chaque session."""
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
    """fit() apprend les vocabulaires sur le train ; transform() construit la matrice de variables."""

    def fit(self, df_train):
        self.vecteurs = {
            famille: CountVectorizer(analyzer=identite, min_df=MIN_DF).fit(docs)
            for famille, docs in jetons_par_famille(df_train).items()}
        return self

    def transform(self, df):
        n_actions = np.maximum(df["actions"].apply(len).to_numpy().astype(float), 1)[:, None]
        navigateur = pd.DataFrame({f"nav__{k}": (df["navigateur"] == k).astype(int) for k in NAVIGATEURS},
                                  index=df.index)
        blocs = [navigateur, variables_globales(df)]
        jetons = jetons_par_famille(df)
        for famille, vecteur in self.vecteurs.items():
            frequences = vecteur.transform(jetons[famille]).toarray() / n_actions   # fréquence relative
            colonnes = [f"{famille}__{t}" for t in vecteur.get_feature_names_out()]
            blocs.append(pd.DataFrame(frequences, index=df.index, columns=colonnes))
        return pd.concat(blocs, axis=1)


# ------------------------------------------------------------------ Debut
class Debut:
    """Indicateurs binaires sur le début de session : 1er écran, 2 premiers écrans, 1ère configuration,
    1ère chaîne, 1ère action et 3 premières actions."""

    @staticmethod
    def _jetons(df):
        sessions = []
        for actions in df["actions"]:
            base = [a[0] for a in actions]
            ecrans = [a[1] for a in actions if a[1]]
            confs = [a[2] for a in actions if a[2]]
            chaines = [a[3] for a in actions if a[3]]
            jetons = []
            if ecrans:
                jetons.append("ecr1=" + ecrans[0])
            if len(ecrans) > 1:
                jetons.append("ecr12=" + ecrans[0] + " > " + ecrans[1])
            if confs:
                jetons.append("conf1=" + confs[0])
            if chaines:
                jetons.append("chn1=" + chaines[0])
            if base:
                jetons.append("act1=" + base[0])
            if len(base) >= 3:
                jetons.append("act123=" + " | ".join(base[:3]))
            sessions.append(jetons)
        return sessions

    def fit(self, df_train):
        self.vecteur = CountVectorizer(analyzer=identite, min_df=MIN_DF, binary=True)
        self.vecteur.fit(self._jetons(df_train))
        return self

    def transform(self, df):
        matrice = self.vecteur.transform(self._jetons(df)).toarray()
        colonnes = ["deb__" + t for t in self.vecteur.get_feature_names_out()]
        return pd.DataFrame(matrice, index=df.index, columns=colonnes)
