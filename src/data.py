"""Étape 1 : lecture et parsing des traces Copilote.

transformation des données brutes en informations structurées

Format brut (sans en-tête, longueur variable) :
    util,navigateur,action1,action2,...,t5,action,...   (util absent dans le test)
Une action peut s'écrire :  Nom(écran)<conf>$chaîne$1
    - (écran)  : écran courant
    - <conf>   : configuration d'écran
    - $chaîne$ : chaîne de la fiche
    - 1 final  : l'utilisateur est en modification
"""
import gzip
import re
from pathlib import Path

import pandas as pd

RE_TEMPS = re.compile(r"t(\d+)")
RE_CHAINE = re.compile(r"\$([^$]*)\$$")
RE_CONF = re.compile(r"<([^>]*)>")
RE_ECRAN = re.compile(r"\(([^)]*)\)")
RE_COLLE = re.compile(r"(.*[^\d\s])t(\d*[05])")   # ex. "Création d'un écrant5" (virgule manquante)


def parse_action(tok):
    """décompose l'action en cinq caractéristiques (base, écran, conf, chaîne, modif)."""
    modif = 0
    if tok.endswith("1"):
        modif, tok = 1, tok[:-1]
    chaine = conf = ecran = None
    m = RE_CHAINE.search(tok)
    if m:
        chaine, tok = m.group(1), tok[:m.start()]
    m = RE_CONF.search(tok)
    if m:
        conf, tok = m.group(1), tok[:m.start()] + tok[m.end():]
    m = RE_ECRAN.search(tok)
    if m:
        ecran, tok = m.group(1), tok[:m.start()] + tok[m.end():]
    return tok.strip().rstrip(")").strip(), ecran, conf, chaine, modif


def parse_ligne(ligne, avec_util):
    """Retourne un dictionnaire : util, navigateur, actions, temps."""
    champs = ligne.rstrip("\r\n").split(",")
    util = champs.pop(0) if avec_util else None
    nav = champs.pop(0) if champs else None
    actions, temps = [], []          # temps = fenêtre (s) de chaque action
    t_courant = 0
    for tok in champs:
        m = RE_TEMPS.fullmatch(tok)
        if m:
            t_courant = int(m.group(1))
        elif tok:
            colle = RE_COLLE.fullmatch(tok)
            if colle:                      # action collée à un marqueur de temps
                actions.append(parse_action(colle.group(1)))
                temps.append(t_courant)
                t_courant = int(colle.group(2))
            else:
                actions.append(parse_action(tok))
                temps.append(t_courant)
    return {"util": util, "navigateur": nav, "actions": actions, "temps": temps}


def charger(chemin, avec_util):
    """Retourne un DataFrame : une ligne = une session."""
    chemin = Path(chemin)
    ouvrir = gzip.open if chemin.suffix.lower() == ".gz" else open
    with ouvrir(chemin, "rt", encoding="utf-8") as f:
        lignes = [parse_ligne(l, avec_util) for l in f if l.strip()]
    df = pd.DataFrame(lignes)
    df.index.name = "session"
    return df


if __name__ == "__main__":
    tr = charger("data/train.csv.GZ", True)
    te = charger("data/test.csv.GZ", False)
    print(tr.shape, te.shape)
    print(tr.iloc[0]["actions"][:5])
