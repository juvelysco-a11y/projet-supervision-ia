"""
06_comparaison.py : comparer les trois méthodes de détection
-------------------------------------------------------------
Méthode 1 : seuil fixe            (comme Zabbix aujourd'hui)
Méthode 2 : z-score glissant      (comparaison avec les 2 dernières heures)
Méthode 3 : Isolation Forest      (IA, qui tient compte de l'heure)

Entrée  : data/reseau_debit.csv
Sorties : tableau comparatif à l'écran, data/comparaison_methodes.csv,
          figures/06_comparaison.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # dessiner dans un fichier, sans fenêtre
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# 1. RÉGLAGES
# ------------------------------------------------------------------
SEUIL = 75               # seuil fixe, en Mbps
FENETRE = 24             # z-score : 24 mesures = 2 heures
SEUIL_Z = 3              # z-score : alerte si |z| > 3
CONTAMINATION = 0.035    # Isolation Forest : part de mesures à signaler

# ------------------------------------------------------------------
# 2. CHARGER ET PRÉPARER
# ------------------------------------------------------------------
df = pd.read_csv("data/reseau_debit.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["heure"] = df["timestamp"].dt.hour
nb_jours = (df["timestamp"].max() - df["timestamp"].min()).days + 1

# ------------------------------------------------------------------
# 3. LES TROIS MÉTHODES (chacune crée une colonne d'alerte 0/1)
# ------------------------------------------------------------------
# Méthode 1 : seuil fixe
df["alerte_seuil"] = (df["debit_mbps"] > SEUIL).astype(int)

# Méthode 2 : z-score glissant
moy = df["debit_mbps"].rolling(FENETRE).mean()
ecart = df["debit_mbps"].rolling(FENETRE).std()
df["z"] = (df["debit_mbps"] - moy) / ecart
df["alerte_z"] = (df["z"].abs() > SEUIL_Z).astype(int)

# Méthode 3 : Isolation Forest (on retire les trous, puis on remet les résultats)
d = df.dropna(subset=["debit_mbps"]).copy()
X = d[["debit_mbps", "heure"]]
modele = IsolationForest(contamination=CONTAMINATION, random_state=42).fit(X)
d["alerte_ia"] = (modele.predict(X) == -1).astype(int)
df["alerte_ia"] = 0
df.loc[d.index, "alerte_ia"] = d["alerte_ia"]     # les trous restent à 0

methodes = {
    "Seuil fixe": "alerte_seuil",
    "Z-score": "alerte_z",
    "Isolation Forest": "alerte_ia",
}

# ------------------------------------------------------------------
# 4. LE BILAN D'UNE MÉTHODE (une fonction réutilisée trois fois)
# ------------------------------------------------------------------
def bilan(colonne):
    vp = ((df[colonne] == 1) & (df["anomalie"] == 1)).sum()
    fp = ((df[colonne] == 1) & (df["anomalie"] == 0)).sum()
    fn = ((df[colonne] == 0) & (df["anomalie"] == 1)).sum()
    precision = vp / max(vp + fp, 1)
    rappel = vp / max(vp + fn, 1)
    f1 = 2 * precision * rappel / max(precision + rappel, 1e-9)
    return {
        "VP": vp, "FP": fp, "FN": fn,
        "Précision": round(precision, 2),
        "Rappel": round(rappel, 2),
        "F1": round(f1, 2),
        "Fausses alarmes/jour": round(fp / nb_jours, 1),
    }


pd.set_option("display.width", 200)          # afficher toutes les colonnes
pd.set_option("display.max_columns", None)

resultats = pd.DataFrame({nom: bilan(col) for nom, col in methodes.items()}).T
resultats[["VP", "FP", "FN"]] = resultats[["VP", "FP", "FN"]].astype(int)
print("=== Bilan par mesure ===")
print(resultats)

# ------------------------------------------------------------------
# 5. LE BILAN PAR ÉVÉNEMENT : détecté ? après combien de minutes ?
# ------------------------------------------------------------------
lignes = []
evenements = []                      # (type, début, fin) pour le graphique
for type_anomalie, g in df[df["anomalie"] == 1].groupby("type_anomalie"):
    debut = g["timestamp"].min()
    fin = g["timestamp"].max()
    evenements.append((type_anomalie, debut, fin))
    ligne = {"Événement": type_anomalie, "Début": debut}
    for nom, col in methodes.items():
        alertes = g[g[col] == 1]
        if alertes.empty:
            ligne[nom] = "raté"
        else:
            minutes = int((alertes["timestamp"].min() - debut).total_seconds() // 60)
            ligne[nom] = f"{minutes} min"
    lignes.append(ligne)

delais = pd.DataFrame(lignes).sort_values("Début")
print()
print("=== Bilan par événement (délai avant la première alerte) ===")
print(delais.to_string(index=False))

# ------------------------------------------------------------------
# 6. SAUVEGARDER LE TABLEAU
# ------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
resultats.to_csv("data/comparaison_methodes.csv")
print()
print("Tableau enregistré : data/comparaison_methodes.csv")

# ------------------------------------------------------------------
# 7. LE GRAPHIQUE : une ligne par méthode, mêmes données, mêmes axes
# ------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

for ax, (nom, col) in zip(axes, methodes.items()):
    # bandes grises = les vraies anomalies (le corrigé)
    for _, debut, fin in evenements:
        ax.axvspan(debut, fin + pd.Timedelta(minutes=5), color="gray", alpha=0.25)
    # courbe du débit
    ax.plot(df["timestamp"], df["debit_mbps"], color="steelblue", lw=0.6)
    # points rouges = alertes de la méthode
    alertes = df[df[col] == 1]
    ax.scatter(alertes["timestamp"], alertes["debit_mbps"], s=7, color="red", zorder=3)
    r = resultats.loc[nom]
    ax.set_title(f"{nom} : précision {r['Précision']}, rappel {r['Rappel']}, "
                 f"{r['Fausses alarmes/jour']} fausse(s) alarme(s)/jour", loc="left")
    ax.set_ylabel("Mbps")

fig.suptitle("Bandes grises = vraies anomalies ; points rouges = alertes de la méthode",
             fontsize=10)
fig.tight_layout()
fig.savefig("figures/06_comparaison.png", dpi=120)
print("Figure enregistrée : figures/06_comparaison.png")
