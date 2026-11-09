#Modèle final

import warnings

import numpy as np
import pandas as pd

from data import DATA, SORTIES, charger_test, charger_train
from modele import Classifieur

warnings.filterwarnings("ignore")


def main():
    train, test = charger_train(), charger_test()
    modele = Classifieur().fit(train, train["util"].to_numpy())
    soumission = pd.DataFrame({"RowId": np.arange(1, len(test) + 1), "prediction": modele.predict(test)})

    modele_fichier = pd.read_csv(DATA / "sample_submission.csv")
    assert list(soumission.columns) == list(modele_fichier.columns), "colonnes de soumission incorrectes"
    assert len(soumission) == len(modele_fichier), "nombre de lignes de soumission incorrect"
    SORTIES.mkdir(exist_ok=True)
    soumission.to_csv(SORTIES / "submission.csv", index=False)
    print(f"{SORTIES / 'submission.csv'} : {len(soumission)} lignes, "
          f"{soumission['prediction'].nunique()} utilisateurs distincts")


if __name__ == "__main__":
    main()
