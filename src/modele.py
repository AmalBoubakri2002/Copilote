"""Étape 3 : modèle final (identification de l'utilisateur à partir d'une session).
Idée : une session est une séquence d'actions. On la représente par des « sacs de n-grammes »
pondérés TF-IDF (actions, écrans, configurations, couples action/écran...), puis on combine
trois modèles complémentaires dont les scores sont normalisés ligne par ligne :
    0,8 x  SVM linéaire (classes équilibrées)   sur les n-grammes TF-IDF + navigateur
    0,1 x  Extra Trees (classes équilibrées)    sur les fréquences relatives (features.py)
    0,1 x  KNN cosinus                          moyenne des 3 meilleures similarités de chaque utilisateur
    -  0,5 x log(fréquence de l'utilisateur dans le train)    correction du biais de fréquence
Poids et coefficient de correction ont été choisis par validation croisée avec simulation de tests
(voir le rapport). Aucun label de validation ou de test n'est utilisé pour apprendre.
"""
from functools import partial

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from features import NAVIGATEURS, Constructeur, Debut

# ---- Paramètres du modèle
POIDS = {"svc": 0.8, "et": 0.1, "knn": 0.1}   # poids des trois composantes dans l'ensemble
BETA = 0.5                                     # force de la correction de fréquence
NAV_POIDS = 3.0                                # le navigateur compte x3 dans le SVM
N_VOISINS = 3                                  # KNN : nombre de similarités moyennées par utilisateur
# (famille de jetons, (n min, n max)) : chaque famille devient un bloc TF-IDF de n-grammes
SPECS = [("base", (1, 3)), ("ecran", (1, 2)), ("conf", (1, 2)),
         ("act_ecran", (1, 2)), ("act_conf_chn", (1, 2)), ("complet", (1, 2))]


# ====================================================================== Représentation TF-IDF
def jetons(actions, famille):
    """Séquence de jetons d'une session pour une famille.
    Une action est un tuple (base, écran, conf, chaîne, modif)."""
    if famille == "base":                       # nom de l'action
        return [a[0] for a in actions]
    if famille == "ecran":                      # écran courant (quand il y en a un)
        return [a[1] for a in actions if a[1]]
    if famille == "conf":                       # configuration d'écran (quand il y en a une)
        return [a[2] for a in actions if a[2]]
    if famille == "act_ecran":                  # couple (action, écran)
        return [f"{a[0]}|{a[1]}" for a in actions]
    if famille == "act_conf_chn":               # couple (configuration, chaîne)
        return [f"{a[2]}|{a[3]}" for a in actions]
    if famille == "complet":                    # action complète à cinq champs
        return [f"{a[0]}|{a[1]}|{a[2]}|{a[3]}|{a[4]}" for a in actions]
    raise ValueError(f"famille inconnue : {famille}")


def ngrammes(actions, famille, taille):
    """n-grammes de jetons consécutifs, pour n allant de taille[0] à taille[1]."""
    seq = jetons(actions, famille)
    resultat = []
    for n in range(taille[0], taille[1] + 1):
        resultat += [" > ".join(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return resultat


class Representation:
    """Représentation d'une session : un bloc TF-IDF par famille de jetons, plus le navigateur.

    Vocabulaire et pondération IDF sont appris sur le train uniquement.
    `transform` renvoie deux matrices : (blocs TF-IDF, navigateur one-hot)."""

    def fit(self, df):
        self.vecteurs = [
            TfidfVectorizer(analyzer=partial(ngrammes, famille=famille, taille=taille),
                            sublinear_tf=True, min_df=2).fit(df["actions"])
            for famille, taille in SPECS]
        return self

    def transform(self, df):
        texte = hstack([v.transform(df["actions"]) for v in self.vecteurs]).tocsr()
        navigateur = np.array([[float(nav == k) for k in NAVIGATEURS] for nav in df["navigateur"]])
        return texte, navigateur

    def noms(self):
        """Nom de chaque colonne de [texte | navigateur] (pour l'importance des variables)."""
        noms = [f"{famille}{taille}::{jeton}" for (famille, taille), v in zip(SPECS, self.vecteurs)
                for jeton in v.get_feature_names_out()]
        return noms + [f"nav::{k}" for k in NAVIGATEURS]


def scores_knn(X_val, X_app, y_app, K):
    """KNN cosinus : pour chaque utilisateur, moyenne de ses N_VOISINS plus fortes similarités avec la session.
    X_val et X_app doivent être normalisées (norme 1)."""
    similarites = (X_val @ X_app.T).toarray()
    scores = np.zeros((similarites.shape[0], K))
    for k in range(K):
        meilleures = -np.sort(-similarites[:, y_app == k], axis=1)[:, :N_VOISINS]
        scores[:, k] = meilleures.mean(axis=1)
    return scores


def zligne(scores):
    """Normalise chaque ligne (moyenne 0, écart-type 1) pour pouvoir combiner des scores d'échelles différentes."""
    return (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-9)


def creer_extra_trees(n_arbres=500, graine=0):
    """Extra Trees utilisé à l'étape 2 (premier modèle) et dans l'ensemble final."""
    return ExtraTreesClassifier(n_estimators=n_arbres, max_features=100, min_samples_leaf=3,
                                class_weight="balanced", n_jobs=-1, random_state=graine)


# ====================================================================== Classifieur
class Classifieur:
    """Ensemble SVM + Extra Trees + KNN cosinus avec correction de la fréquence des utilisateurs."""

    def __init__(self, n_arbres=500, graine=0):
        self.n_arbres, self.graine = n_arbres, graine

    # ---------------------------------------------------------------- apprentissage
    def fit(self, df, y):
        self.classes = np.array(sorted(set(y)))
        self.y = np.searchsorted(self.classes, y)                      # labels entiers 0..K-1
        self.K = len(self.classes)
        self.log_prior = np.log(np.bincount(self.y, minlength=self.K) / len(self.y))

        # SVM et KNN : n-grammes TF-IDF
        self.representation = Representation().fit(df)
        X_svm, X_cos = self._matrices(df)
        self.svc = LinearSVC(C=1.0, class_weight="balanced").fit(X_svm, self.y)
        self.X_cos = X_cos

        # Extra Trees : fréquences relatives
        self.constructeur, self.debut = Constructeur().fit(df), Debut().fit(df)
        self.et = creer_extra_trees(self.n_arbres, self.graine).fit(self._frequences(df), self.y)
        return self

    def _matrices(self, df):
        """(matrice du SVM : TF-IDF + navigateur pondéré,  matrice du KNN : TF-IDF normalisé sans navigateur)."""
        texte, navigateur = self.representation.transform(df)
        X_svm = hstack([texte, csr_matrix(NAV_POIDS * navigateur)]).tocsr()
        return X_svm, normalize(texte).tocsr()

    def _frequences(self, df):
        return pd.concat([self.constructeur.transform(df), self.debut.transform(df)], axis=1)

    # ---------------------------------------------------------------- prédiction
    def composantes(self, df):
        """Scores bruts des trois modèles : {"svc", "et", "knn"}, chacun de forme (n_sessions, K)."""
        X_svm, X_cos = self._matrices(df)
        return {"svc": self.svc.decision_function(X_svm),
                "et": self.et.predict_proba(self._frequences(df)),
                "knn": scores_knn(X_cos, self.X_cos, self.y, self.K)}

    def combiner(self, composantes):
        """Ensemble pondéré des scores normalisés, moins la correction de fréquence."""
        ensemble = sum(POIDS[nom] * zligne(composantes[nom]) for nom in POIDS)
        return ensemble - BETA * self.log_prior[None, :]

    def predict(self, df):
        return self.classes[self.combiner(self.composantes(df)).argmax(1)]

    # ---------------------------------------------------------------- interprétation
    def importance(self, top=10):
        """Importance des variables : |coefficient| moyen du SVM sur les utilisateurs.
        Retourne (part de chaque famille, top variables hors navigateur)."""
        noms = np.array(self.representation.noms())
        importance = np.abs(self.svc.coef_).mean(axis=0)
        familles = pd.Series(importance, index=[n.split("::")[0] for n in noms]).groupby(level=0).sum()
        familles = (familles / familles.sum()).sort_values(ascending=False)
        ordre = [i for i in np.argsort(importance)[::-1] if not noms[i].startswith("nav::")][:top]
        return familles, [(noms[i], float(importance[i])) for i in ordre]
