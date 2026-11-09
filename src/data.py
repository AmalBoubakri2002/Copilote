"""Étape 1 : lecture et décomposition des traces Copilote.

Une ligne du fichier = une session, sans en-tête, de longueur variable :
   une action s'écrit  Nom(écran)<configuration>$chaîne$1  où chaque partie est facultative :
        (écran)      écran courant
        <conf>       configuration de l'écran
        $chaîne$     chaîne de la fiche
        1 final      l'utilisateur est en modification
"""
import gzip
import re
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
DATA, SORTIES = RACINE / "data", RACINE / "outputs"

RE_TEMPS = re.compile(r"t(\d+)")
RE_CHAINE = re.compile(r"\$([^$]*)\$$")
RE_CONF = re.compile(r"<([^>]*)>")
RE_ECRAN = re.compile(r"\(([^)]*)\)")
# Action collée à un marqueur de temps, ex. 
RE_COLLE = re.compile(r"(.*[^\d\s])t(\d*[05])")


def parse_action(jeton):
    """Décompose une action en (base, écran, conf, chaîne, modif)."""
    modif = 0
    if jeton.endswith("1"):
        modif, jeton = 1, jeton[:-1]
    chaine = conf = ecran = None
    m = RE_CHAINE.search(jeton)
    if m:
        chaine, jeton = m.group(1), jeton[:m.start()]
    m = RE_CONF.search(jeton)
    if m:
        conf, jeton = m.group(1), jeton[:m.start()] + jeton[m.end():]
    m = RE_ECRAN.search(jeton)
    if m:
        ecran, jeton = m.group(1), jeton[:m.start()] + jeton[m.end():]
    base = jeton.strip().rstrip(")").strip()
    return base, ecran, conf, chaine, modif


def parse_ligne(ligne, avec_util):
    """Transforme une ligne en dictionnaire : util, navigateur, actions, temps.

    `temps[i]` est la fenêtre de temps (en secondes) dans laquelle a lieu l'action i."""
    champs = ligne.rstrip("\r\n").split(",")
    util = champs.pop(0) if avec_util else None
    navigateur = champs.pop(0) if champs else None
    actions, temps = [], []
    t_courant = 0
    for jeton in champs:
        marqueur = RE_TEMPS.fullmatch(jeton)
        if marqueur:
            t_courant = int(marqueur.group(1))
        elif jeton:
            colle = RE_COLLE.fullmatch(jeton)
            if colle:                                  # action suivie d'un marqueur de temps
                actions.append(parse_action(colle.group(1)))
                temps.append(t_courant)
                t_courant = int(colle.group(2))
            else:
                actions.append(parse_action(jeton))
                temps.append(t_courant)
    return {"util": util, "navigateur": navigateur, "actions": actions, "temps": temps}


def charger(chemin, avec_util):
    """Lit un fichier de traces (.csv ou .gz) et retourne un DataFrame : une ligne = une session."""
    chemin = Path(chemin)
    ouvrir = gzip.open if chemin.suffix.lower() == ".gz" else open
    with ouvrir(chemin, "rt", encoding="utf-8") as f:
        sessions = [parse_ligne(ligne, avec_util) for ligne in f if ligne.strip()]
    df = pd.DataFrame(sessions)
    df.index.name = "session"
    return df


def charger_train(garder_vides=False):
    """Sessions d'entraînement. Par défaut, la session vide (aucune action), qui ne porte
    aucune information, est supprimée."""
    train = charger(DATA / "train.csv.GZ", avec_util=True)
    return train if garder_vides else train[train["actions"].apply(len) > 0].reset_index(drop=True)


def charger_test():
    """Sessions de test (jamais filtrées : 324 lignes attendues pour la soumission)."""
    return charger(DATA / "test.csv.GZ", avec_util=False)
