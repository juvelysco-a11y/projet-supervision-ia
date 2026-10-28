"""
07_test_donnees_reelles.py : les trois méthodes sur de VRAIES données réseau
---------------------------------------------------------------------------
Jeu de données : NAB (Numenta Anomaly Benchmark), trafic réseau entrant
d'un serveur réel (octets reçus toutes les 5 minutes), avec des anomalies
étiquetées par des humains.

Nouveauté par rapport au script 06 : SÉPARATION ENTRAÎNEMENT / TEST
  - ENTRAÎNEMENT = la période AVANT la première anomalie connue.
    C'est là que chaque méthode "apprend le normal" (réglage du seuil,
    entraînement de l'IA).
  - TEST = tout ce qui vient APRÈS. On note les méthodes seulement ici,
    sur des mesures qu'elles n'ont jamais vues.

Entrées : data/nab_labels.json  et  data/<FICHIER>.csv
Sortie  : bilan à l'écran + figures/07_donnees_reelles.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# 1. RÉGLAGES
# ------------------------------------------------------------------
NOM_FICHIER = "ec2_network_in_5abac7"    # essaie aussi : ec2_network_in_257a54
FENETRE = 24                             # z-score : 24 mesures = 2 heures
SEUIL_Z = 3
CONTAMINATION = 0.01                     # IA : part de l'ENTRAÎNEMENT jugée "bizarre"
QUANTILE_SEUIL = 0.99                    # seuil fixe : 99 % des valeurs normales en dessous

# ------------------------------------------------------------------
# 2. CHARGER LES DONNÉES ET LES ÉTIQUETTES
# ------------------------------------------------------------------
df = pd.read_csv(f"data/{NOM_FICHIER}.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.rename(columns={"value": "valeur"})
df["heure"] = df["timestamp"].dt.hour

# NAB donne les anomalies sous forme de FENÊTRES [début, fin] de temps.
with open("data/nab_labels.json") as f:
    fenetres = json.load(f)[f"realAWSCloudwatch/{NOM_FICHIER}.csv"]

df["anomalie"] = 0
for debut, fin in fenetres:
    dans_fenetre = (df["timestamp"] >= debut) & (df["timestamp"] <= fin)
    df.loc[dans_fenetre, "anomalie"] = 1

print(f"{len(df)} mesures, dont {df['anomalie'].sum()} dans une fenêtre d'anomalie")

# ------------------------------------------------------------------
# 3. SÉPARER ENTRAÎNEMENT / TEST
# ------------------------------------------------------------------
premiere_anomalie = df.index[df["anomalie"] == 1][0]    # numéro de la 1re ligne anormale
train = df.loc[:premiere_anomalie - 1]                  # avant
test = df.loc[premiere_anomalie:].copy()                # à partir de la 1re anomalie
print(f"Entraînement : {len(train)} mesures (aucune anomalie connue)")
print(f"Test         : {len(test)} mesures, dont {test['anomalie'].sum()} anormales")

# ------------------------------------------------------------------
# 4. LES TROIS MÉTHODES
# ------------------------------------------------------------------
# Méthode 1 : seuil fixe, réglé sur l'entraînement.
# On ne choisit plus 75 "au hasard" : on prend la valeur sous laquelle
# 99 % des mesures NORMALES de l'entraînement se trouvent.
seuil = train["valeur"].quantile(QUANTILE_SEUIL)
test["alerte_seuil"] = (test["valeur"] > seuil).astype(int)
print(f"Seuil fixe choisi sur l'entraînement : {seuil:,.0f}")

# Méthode 2 : z-score glissant (calculé sur toute la série, car il ne
# regarde que les 2 dernières heures : il n'a rien à apprendre).
moy = df["valeur"].rolling(FENETRE).mean()
ecart = df["valeur"].rolling(FENETRE).std()
z = (df["valeur"] - moy) / ecart
test["alerte_z"] = (z.loc[test.index].abs() > SEUIL_Z).astype(int)

# Méthode 3 : Isolation Forest, entraînée UNIQUEMENT sur la période d'entraînement.
# Les valeurs vont de quelques dizaines à des millions : on passe au
# logarithme (log1p) pour que l'IA puisse comparer des ordres de grandeur.
def preparer(tableau):
    return pd.DataFrame({
        "log_valeur": np.log1p(tableau["valeur"]),
        "heure": tableau["heure"],
    })

modele = IsolationForest(contamination=CONTAMINATION, random_state=42)
modele.fit(preparer(train))                              # apprend le normal (train)
test["alerte_ia"] = (modele.predict(preparer(test)) == -1).astype(int)   # juge le test

methodes = {
    "Seuil fixe": "alerte_seuil",
    "Z-score": "alerte_z",
    "Isolation Forest": "alerte_ia",
}

# ------------------------------------------------------------------
# 5. LE BILAN (sur le TEST uniquement)
# ------------------------------------------------------------------
nb_jours = (test["timestamp"].max() - test["timestamp"].min()).days + 1


def bilan(colonne):
    # Une fenêtre NAB est une ZONE DE TOLÉRANCE autour d'un événement, pas une
    # période où chaque mesure est anormale. On note donc PAR FENÊTRE :
    #   fenêtre détectée = au moins une alerte dedans.
    detectees = 0
    for debut, fin in fenetres:
        morceau = test[(test["timestamp"] >= debut) & (test["timestamp"] <= fin)]
        if (morceau[colonne] == 1).any():
            detectees += 1
    alertes_dans = ((test[colonne] == 1) & (test["anomalie"] == 1)).sum()
    fausses = ((test[colonne] == 1) & (test["anomalie"] == 0)).sum()   # hors fenêtre
    total = alertes_dans + fausses
    return {"Fenêtres détectées": f"{detectees} sur {len(fenetres)}",
            "Alertes dans une fenêtre": alertes_dans,
            "Fausses alarmes (hors fenêtre)": fausses,
            "Précision": round(alertes_dans / max(total, 1), 2),
            "Fausses alarmes/jour": round(fausses / nb_jours, 1)}


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
resultats = pd.DataFrame({nom: bilan(col) for nom, col in methodes.items()}).T
print()
print("=== Bilan sur la période de TEST (noté par fenêtre) ===")
print(resultats)

# Bilan par fenêtre d'anomalie : vue ? après combien de minutes ?
print()
print("=== Par fenêtre d'anomalie (délai avant la première alerte) ===")
for numero, (debut, fin) in enumerate(fenetres, start=1):
    morceau = test[(test["timestamp"] >= debut) & (test["timestamp"] <= fin)]
    ligne = f"Fenêtre {numero} ({debut[:16]} -> {fin[:16]}) : "
    for nom, col in methodes.items():
        alertes = morceau[morceau[col] == 1]
        if alertes.empty:
            ligne += f"{nom} = raté ; "
        else:
            minutes = int((alertes["timestamp"].min() - pd.Timestamp(debut)).total_seconds() // 60)
            ligne += f"{nom} = {max(minutes, 0)} min ; "
    print(ligne)

# ------------------------------------------------------------------
# 6. LE GRAPHIQUE (échelle logarithmique : sinon on ne voit que les pics)
# ------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (nom, col) in zip(axes, methodes.items()):
    for debut, fin in fenetres:
        ax.axvspan(pd.Timestamp(debut), pd.Timestamp(fin), color="gray", alpha=0.25)
    ax.plot(test["timestamp"], test["valeur"], color="steelblue", lw=0.6)
    alertes = test[test[col] == 1]
    ax.scatter(alertes["timestamp"], alertes["valeur"], s=7, color="red", zorder=3)
    ax.set_yscale("log")
    r = resultats.loc[nom]
    ax.set_title(f"{nom} : {r['Fenêtres détectées']} fenêtres détectées, "
                 f"{r['Fausses alarmes/jour']} fausse(s) alarme(s)/jour", loc="left")
    ax.set_ylabel("octets / 5 min (log)")
fig.suptitle("Données RÉELLES (NAB) - période de test. Bandes grises = anomalies étiquetées",
             fontsize=10)
fig.tight_layout()
fig.savefig("figures/07_donnees_reelles.png", dpi=120)
print("\nFigure enregistrée : figures/07_donnees_reelles.png")
