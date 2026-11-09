"""ÉTAPE 3 : optimisation de la solution de classification automatique.
Démarche : on part du premier modèle de l'étape 2 (Extra Trees, F1 macro 0,9515) et on change UNE chose à la fois,
en gardant ce qui améliore le F1 macro. Toutes les étapes sont comparées sur exactement les mêmes plis
(validation croisée stratifiée en 4 plis, graine 0 : mêmes plis que l'étape 2).
    Partie A  Changement de représentation : n-grammes TF-IDF + SVM linéaire, enrichis famille par famille
    Partie B  Poids du navigateur, pondération des classes
    Partie C  Ensemble SVM + Extra Trees + KNN cosinus, puis correction du biais de fréquence
              (évaluée aussi sur des tests simulés de 324 sessions, voir plus bas)
Fin du script : variables les plus influentes du modèle final (coefficients du SVM).
"""
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          
import numpy as np                     
from scipy.sparse import csr_matrix, hstack            
from sklearn.feature_extraction.text import TfidfVectorizer      
from sklearn.metrics import accuracy_score, f1_score     
from sklearn.model_selection import StratifiedKFold     
from sklearn.preprocessing import normalize             
from sklearn.svm import LinearSVC                       

from data import SORTIES, charger_train                 
from features import NAVIGATEURS                       
from modele import BETA, NAV_POIDS, POIDS, SPECS, Classifieur, ngrammes, scores_knn, zligne    

GRAINE, N_PLIS = 0, 4
CACHE = SORTIES / "cache"
TOUTES = [famille for famille, _ in SPECS]               # base, ecran, conf, act_ecran, act_conf_chn, complet

# Partie A et B : chaque configuration = (familles de jetons, poids du navigateur, pondération des classes)
CONFIGURATIONS = {
    "A1  n-grammes d'actions (1 à 3) + SVM":                    (["base"], 1.0, None),
    "A2  + écrans et configurations":                           (["base", "ecran", "conf"], 1.0, None),
    "A3  + couples action|écran et configuration|chaîne":       (["base", "ecran", "conf", "act_ecran", "act_conf_chn"], 1.0, None),
    "A4  + action complète (5 champs)":                         (TOUTES, 1.0, None),
    "B1  navigateur pondéré x3":                                (TOUTES, NAV_POIDS, None),
    "B2  classes équilibrées (= SVM final)":                    (TOUTES, NAV_POIDS, "balanced"),
}
SVM_FINAL = "B2  classes équilibrées (= SVM final)"


def titre(texte):
    print(f"\n{'=' * 100}\n{texte}\n{'=' * 100}")


# ====================================================================== calcul par pli (avec cache)
def calculer_pli(train, y, K, numero, apprentissage, validation):
    """Scores de validation de toutes les configurations et du KNN pour un pli."""
    fichier = CACHE / f"pli{numero}.npz"
    if fichier.exists():
        return dict(np.load(fichier))
    d_app, d_val = train.iloc[apprentissage], train.iloc[validation]
    blocs_app, blocs_val = {}, {}
    for famille, taille in SPECS:                     # un bloc TF-IDF par famille, appris sur le pli seulement
        tfidf = TfidfVectorizer(analyzer=partial(ngrammes, famille=famille, taille=taille),
                                sublinear_tf=True, min_df=2).fit(d_app["actions"])
        blocs_app[famille], blocs_val[famille] = tfidf.transform(d_app["actions"]), tfidf.transform(d_val["actions"])
    nav = lambda d: np.array([[float(n == k) for k in NAVIGATEURS] for n in d["navigateur"]])      # noqa: E731
    nav_app, nav_val = nav(d_app), nav(d_val)

    sorties = {"validation": validation, "prior": np.bincount(y[apprentissage], minlength=K)}
    for nom, (familles, poids_nav, poids_classes) in CONFIGURATIONS.items():
        X_app = hstack([blocs_app[f] for f in familles] + [csr_matrix(poids_nav * nav_app)]).tocsr()
        X_val = hstack([blocs_val[f] for f in familles] + [csr_matrix(poids_nav * nav_val)]).tocsr()
        svm = LinearSVC(C=1.0, class_weight=poids_classes).fit(X_app, y[apprentissage])
        sorties[nom] = svm.decision_function(X_val).astype(np.float32)
        print(f"   pli {numero} : {nom[:3]} terminé", flush=True)
    cos_app = normalize(hstack([blocs_app[f] for f in TOUTES]).tocsr())
    cos_val = normalize(hstack([blocs_val[f] for f in TOUTES]).tocsr())
    sorties["knn"] = scores_knn(cos_val, cos_app, y[apprentissage], K).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fichier, **sorties)
    return sorties


# ====================================================================== métriques
def resume(y, scores, K, rares):
    """F1 macro, accuracy et F1 des utilisateurs rares à partir d'une matrice de scores (n, K)."""
    pred = scores.argmax(1)
    f1_classes = f1_score(y, pred, average=None, labels=np.arange(K))
    return {"f1": f1_score(y, pred, average="macro"), "acc": accuracy_score(y, pred), "rares": f1_classes[rares].mean()}


def f1_test_simule(y, pred, K, n_tirages=300, n_test=324, graine=0):
    """F1 macro moyen sur des tests simulés : à chaque tirage, un découpage stratifié proportionnel de n_test sessions
    (nombre de sessions par utilisateur ~ proportionnel à sa taille, arrondi aléatoire), avec la convention du
    F1 macro sur l'union des classes vraies et prédites. Les prédictions viennent de la validation croisée, donc
    aucune session n'est jamais prédite par un modèle qui l'a vue. Même tirages pour tous les modèles (graine fixe)."""
    rng = np.random.default_rng(graine)
    par_util = [np.where(y == k)[0] for k in range(K)]
    a = n_test / len(y)
    valeurs = []
    for _ in range(n_tirages):
        attendu = a * np.array([len(i) for i in par_util])
        k_u = np.floor(attendu).astype(int) + (rng.random(K) < attendu - np.floor(attendu))
        idx = np.concatenate([rng.choice(i, min(k, len(i)), replace=False) for i, k in zip(par_util, k_u) if k > 0])
        yt, yp = y[idx], pred[idx]
        tp = np.bincount(yt[yt == yp], minlength=K)
        den = np.bincount(yt, minlength=K) + np.bincount(yp, minlength=K)
        valeurs.append((2 * tp[den > 0] / den[den > 0]).mean())
    return float(np.mean(valeurs))


# ====================================================================== programme principal
train = charger_train()
classes = np.array(sorted(train["util"].unique()))
y = np.searchsorted(classes, train["util"].to_numpy())
K = len(classes)
rares = np.bincount(y, minlength=K) <= 5
try:
    et = np.load(CACHE / "et_oof.npy")
except FileNotFoundError:
    raise SystemExit("Scores de l'Extra Trees introuvables : lancer d'abord `python src/etape2_premier_modele.py`.")

titre("Calcul des scores de validation (4 plis ; mis en cache, donc reprenable)")
plis = list(StratifiedKFold(N_PLIS, shuffle=True, random_state=GRAINE).split(train, y))
sorties = [calculer_pli(train, y, K, i, a, v) for i, (a, v) in enumerate(plis)]

n = len(y)
S = {nom: np.zeros((n, K), dtype=np.float32) for nom in [*CONFIGURATIONS, "knn"]}
log_prior = np.zeros((n, K), dtype=np.float32)
for s in sorties:                                           # recollage des scores de validation de chaque pli
    for nom in S:
        S[nom][s["validation"]] = s[nom]
    log_prior[s["validation"]] = np.log(s["prior"] / s["prior"].sum())[None, :]

titre("Parties A et B : représentation, navigateur, pondération des classes (SVM, mêmes plis)  |  Partie C : ensemble")
z_svm, z_et, z_knn = zligne(S[SVM_FINAL]), zligne(et), zligne(S["knn"])
c1 = 0.8 * z_svm + 0.2 * z_et
c2 = POIDS["svc"] * z_svm + POIDS["et"] * z_et + POIDS["knn"] * z_knn
c3 = c2 - BETA * log_prior                                  # modèle final (voir Classifieur.combiner)
scores_modeles = {
    "Étape 2  Extra Trees (fréquences relatives)": et,
    **{nom: S[nom] for nom in CONFIGURATIONS},
    "C1  SVM + Extra Trees (0,8 / 0,2)": c1,
    "C2  + KNN cosinus (poids 0,8 / 0,1 / 0,1)": c2,
    f"C3  + correction du biais de fréquence (beta = {BETA})": c3,
}
resultats = {nom: resume(y, sc, K, rares) for nom, sc in scores_modeles.items()}
simule = {nom: f1_test_simule(y, sc.argmax(1), K) for nom, sc in scores_modeles.items()}      # mêmes tirages pour tous

print(f"{'modèle':66s} {'F1 macro':>9s} {'accuracy':>9s} {'F1 rares':>9s} {'F1 test simulé':>15s}")
for nom, r in resultats.items():
    print(f"{nom:66s} {r['f1']:9.4f} {r['acc']:9.4f} {r['rares']:9.4f} {simule[nom]:15.4f}")
print("\nF1 macro, accuracy et F1 rares : validation croisée stratifiée 4 plis (utilisateurs rares : 5 sessions ou moins).")
print("F1 test simulé : moyenne sur 300 tirages de 324 sessions (découpage stratifié proportionnel), F1 macro sur l'union des classes.")
print("Poids de l'ensemble et beta : choisis lors de l'étude complète par validation croisée avec simulation de tests ;"
      " ce tableau les compare sur les mêmes plis.")

# ------------------------------------------------------------------ figure
(SORTIES / "figures").mkdir(exist_ok=True)
noms = list(resultats)
fig, ax = plt.subplots(figsize=(9, 4.2))
valeurs = [resultats[n]["f1"] for n in noms]
couleurs = ["#9aa5b1"] + ["#4c78a8"] * len(CONFIGURATIONS) + ["#2a9d8f"] * 3
ax.barh(range(len(noms))[::-1], valeurs, color=couleurs)
ax.set_yticks(range(len(noms))[::-1])
ax.set_yticklabels([n[:62] for n in noms], fontsize=7.5)
for i, v in zip(range(len(noms))[::-1], valeurs):
    ax.text(v + 0.0008, i, f"{v:.4f}", va="center", fontsize=8)
ax.set(xlim=(0.90, 0.985), xlabel="F1 macro (validation croisée, 4 plis)", title="Étape 3 : progression du F1 macro")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(SORTIES / "figures" / "etape3_progression.png", dpi=150)
print("\nfigure enregistrée : outputs/figures/etape3_progression.png")

# ------------------------------------------------------------------ variables les plus influentes
titre("Variables les plus influentes du modèle final (|coefficient| moyen du SVM sur les 247 utilisateurs)")
modele = Classifieur(n_arbres=10).fit(train, train["util"].to_numpy())      # 10 arbres : l'Extra Trees n'est pas utilisé ici
familles, top = modele.importance()
print("Part de l'importance par famille de variables :")
print(familles.round(3).to_string())
print("\nTop 10 variables :")
for nom, valeur in top:
    print(f"   {valeur:.4f}  {nom[:100]}")
