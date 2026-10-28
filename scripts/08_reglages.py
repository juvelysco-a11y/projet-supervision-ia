"""
08_reglages.py : réduire les fausses alarmes en modifiant les réglages
----------------------------------------------------------------------
Idée : chaque méthode a un "bouton" de sensibilité.
    Seuil fixe       -> le quantile        (0.99, 0.995, 0.999 ...)
    Z-score          -> le seuil de z      (3, 4, 5, 6 ...)
    Isolation Forest -> la contamination   (0.01, 0.005, 0.001 ...)
On tourne chaque bouton et on regarde l'effet sur 2 chiffres :
    - combien de fenêtres d'anomalie sont encore détectées ?
    - combien de fausses alarmes par jour ?

Deux nouveautés :
    1. on le fait sur les TROIS jeux de données d'un coup (boucle)
    2. on compte aussi les INCIDENTS : des alertes très proches dans le temps
       (moins de 60 min d'écart) comptent pour UNE seule notification,
       comme le ferait un vrai outil de supervision.

Entrées : data/nab_labels.json et les 3 fichiers .csv dans data/
Sortie  : tableaux à l'écran + data/comparaison_reglages.csv
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# 1. RÉGLAGES DE L'EXPÉRIENCE
# ------------------------------------------------------------------
JEUX = [
    "ec2_network_in_5abac7",
    "ec2_network_in_257a54",
    "iio_us-east-1_i-a2eb1cd9_NetworkIn",
]
FENETRE = 24                          # z-score : 24 mesures = 2 heures
QUANTILES = [0.99, 0.995, 0.999]      # boutons du seuil fixe
SEUILS_Z = [3, 4, 5, 6]               # boutons du z-score
CONTAMINATIONS = [0.01, 0.005, 0.001]  # boutons de l'IA
PAUSE_INCIDENT = pd.Timedelta(minutes=60)   # alertes plus proches = même incident

with open("data/nab_labels.json") as f:
    tous_les_labels = json.load(f)


# ------------------------------------------------------------------
# 2. CHARGER UN JEU ET LE SÉPARER EN ENTRAÎNEMENT / TEST
#    (même logique que le script 07, rangée dans une fonction)
# ------------------------------------------------------------------
def charger(nom):
    df = pd.read_csv(f"data/{nom}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"value": "valeur"})
    df["heure"] = df["timestamp"].dt.hour

    fenetres = tous_les_labels[f"realAWSCloudwatch/{nom}.csv"]
    df["anomalie"] = 0
    for debut, fin in fenetres:
        dans = (df["timestamp"] >= debut) & (df["timestamp"] <= fin)
        df.loc[dans, "anomalie"] = 1

    premiere = df.index[df["anomalie"] == 1][0]
    train = df.loc[:premiere - 1]
    test = df.loc[premiere:].copy()
    return df, fenetres, train, test


def preparer(tableau):
    """Ce qu'on donne à l'IA : le logarithme de la valeur, et l'heure."""
    return pd.DataFrame({
        "log_valeur": np.log1p(tableau["valeur"]),
        "heure": tableau["heure"],
    })


# ------------------------------------------------------------------
# 3. NOTER UNE COLONNE D'ALERTES
# ------------------------------------------------------------------
def noter(test, fenetres, colonne):
    nb_jours = (test["timestamp"].max() - test["timestamp"].min()).days + 1

    # fenêtres détectées : au moins une alerte dedans
    detectees = 0
    for debut, fin in fenetres:
        morceau = test[(test["timestamp"] >= debut) & (test["timestamp"] <= fin)]
        if (morceau[colonne] == 1).any():
            detectees += 1

    # fausses alarmes = alertes hors de toute fenêtre
    fausses = test[(test[colonne] == 1) & (test["anomalie"] == 0)]

    # incidents : on regroupe les fausses alarmes proches dans le temps.
    # Un nouvel incident commence quand l'écart avec l'alerte précédente
    # dépasse PAUSE_INCIDENT.
    heures = fausses["timestamp"].sort_values()
    nouveaux = heures.diff() > PAUSE_INCIDENT          # True = début d'un incident
    nb_incidents = int(nouveaux.sum()) + (1 if len(heures) > 0 else 0)

    return {
        "Fenêtres détectées": f"{detectees}/{len(fenetres)}",
        "Fausses alarmes": len(fausses),
        "Par jour": round(len(fausses) / nb_jours, 1),
        "Incidents faux": nb_incidents,
        "Incidents/jour": round(nb_incidents / nb_jours, 1),
    }


# ------------------------------------------------------------------
# 4. LES EXPÉRIENCES : on tourne chaque bouton, pour chaque jeu
# ------------------------------------------------------------------
lignes = []

for nom in JEUX:
    df, fenetres, train, test = charger(nom)

    # --- seuil fixe : on tourne le quantile
    for q in QUANTILES:
        seuil = train["valeur"].quantile(q)
        test["alerte"] = (test["valeur"] > seuil).astype(int)
        r = noter(test, fenetres, "alerte")
        lignes.append({"Jeu": nom[-12:], "Méthode": "Seuil fixe",
                       "Réglage": f"quantile {q}", **r})

    # --- z-score : on tourne le seuil de z (le calcul de z ne change pas)
    moy = df["valeur"].rolling(FENETRE).mean()
    ecart = df["valeur"].rolling(FENETRE).std()
    z = ((df["valeur"] - moy) / ecart).loc[test.index]
    for sz in SEUILS_Z:
        test["alerte"] = (z.abs() > sz).astype(int)
        r = noter(test, fenetres, "alerte")
        lignes.append({"Jeu": nom[-12:], "Méthode": "Z-score",
                       "Réglage": f"seuil z {sz}", **r})

    # --- Isolation Forest : on tourne la contamination
    for c in CONTAMINATIONS:
        modele = IsolationForest(contamination=c, random_state=42)
        modele.fit(preparer(train))
        test["alerte"] = (modele.predict(preparer(test)) == -1).astype(int)
        r = noter(test, fenetres, "alerte")
        lignes.append({"Jeu": nom[-12:], "Méthode": "Isolation Forest",
                       "Réglage": f"contamination {c}", **r})

resultats = pd.DataFrame(lignes)

# ------------------------------------------------------------------
# 5. AFFICHAGE : un tableau par jeu de données
# ------------------------------------------------------------------
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

for jeu, tableau in resultats.groupby("Jeu", sort=False):
    print()
    print(f"=== Jeu ...{jeu} ===")
    print(tableau.drop(columns="Jeu").to_string(index=False))

resultats.to_csv("data/comparaison_reglages.csv", index=False)
print()
print("Tableau enregistré : data/comparaison_reglages.csv")
