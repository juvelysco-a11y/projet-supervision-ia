"""
Générateur de données v1 : débit réseau entrant d'un lien (en Mbps)
-------------------------------------------------------------------
- 4 semaines de mesures, 1 mesure toutes les 5 minutes
- profil journalier réaliste (creux la nuit, pics en journée, week-end calme)
- tendance lente + bruit + valeurs manquantes
- 5 types d'anomalies connues à l'avance (vérité terrain)

Résultat : data/reseau_debit.csv avec les colonnes
    timestamp, debit_mbps, anomalie (0/1), type_anomalie
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. PARAMÈTRES (on les met en haut pour pouvoir les modifier facilement)
# ------------------------------------------------------------------
GRAINE = 42                  # fixe le hasard : mêmes données à chaque exécution
DEBUT = "2026-09-21"         # un lundi, 00:00
NB_JOURS = 28                # 4 semaines
PAS = "5min"                 # une mesure toutes les 5 minutes
CAPACITE_MBPS = 100          # capacité maximale du lien (on ne peut pas dépasser)

rng = np.random.default_rng(GRAINE)   # générateur aléatoire reproductible

# ------------------------------------------------------------------
# 2. L'AXE DU TEMPS
# ------------------------------------------------------------------
temps = pd.date_range(start=DEBUT, periods=NB_JOURS * 288, freq=PAS)
# 288 = nombre de mesures par jour (24 h x 12 mesures par heure)

df = pd.DataFrame({"timestamp": temps})
heure = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60   # ex : 14.5 = 14h30
jour_semaine = df["timestamp"].dt.dayofweek                        # 0 = lundi ... 6 = dimanche
jour_ecoule = np.arange(len(df)) / 288                             # nombre de jours écoulés

# ------------------------------------------------------------------
# 3. LE PROFIL NORMAL (sans anomalie)
# ------------------------------------------------------------------
# Deux "bosses" : une le matin vers 11h, une l'après-midi vers 15h30
bosse_matin = np.exp(-((heure - 11) / 3) ** 2)
bosse_aprem = 0.9 * np.exp(-((heure - 15.5) / 2.5) ** 2)
activite = bosse_matin + bosse_aprem

# Le week-end (samedi=5, dimanche=6), l'activité est réduite à 35 %
facteur_semaine = np.where(jour_semaine >= 5, 0.35, 1.0)

base = 15 + 40 * activite * facteur_semaine      # 15 Mbps la nuit, ~55 Mbps au pic

# Tendance : le trafic augmente lentement (+0,4 % par jour)
tendance = 1 + 0.004 * jour_ecoule

signal = base * tendance

# Bruit : variations aléatoires normales, proportionnelles au signal (5 %)
bruit = rng.normal(0, 0.05 * signal)
debit = signal + bruit

# ------------------------------------------------------------------
# 4. LES ANOMALIES (vérité terrain)
# ------------------------------------------------------------------
df["anomalie"] = 0
df["type_anomalie"] = ""


def indices(jour, heure_debut, duree_min):
    """Renvoie les positions (indices) des mesures d'une fenêtre donnée."""
    debut = int(jour * 288 + heure_debut * 12)
    fin = debut + duree_min // 5
    return np.arange(debut, fin)


def marquer(idx, nom):
    """Note dans le tableau que ces mesures sont anormales."""
    df.loc[idx, "anomalie"] = 1
    df.loc[idx, "type_anomalie"] = nom


# (a) PIC BRUTAL : +60 Mbps pendant 1 h, vendredi de la semaine 1 (jour 4), 10h00
idx = indices(jour=4, heure_debut=10, duree_min=60)
debit[idx] += 60
marquer(idx, "pic_brutal")

# (b) DÉRIVE PROGRESSIVE : montée lente de 0 à +45 Mbps sur 6 h, jeudi semaine 2 (jour 10), 13h00
idx = indices(jour=10, heure_debut=13, duree_min=360)
debit[idx] += np.linspace(0, 45, len(idx))
marquer(idx, "derive_progressive")

# (c) CHANGEMENT DE NIVEAU : +25 Mbps de façon durable pendant 12 h, lundi semaine 3 (jour 14), 06h00
idx = indices(jour=14, heure_debut=6, duree_min=720)
debit[idx] += 25
marquer(idx, "changement_niveau")

# (d) CHUTE À ZÉRO : coupure de lien de 45 min, vendredi semaine 3 (jour 18), 14h00
idx = indices(jour=18, heure_debut=14, duree_min=45)
debit[idx] = rng.uniform(0, 0.5, len(idx))
marquer(idx, "chute_zero")

# (e) VARIANCE QUI AUGMENTE : trafic instable pendant 4 h, mercredi semaine 4 (jour 23), 09h00
idx = indices(jour=23, heure_debut=9, duree_min=240)
debit[idx] += rng.normal(0, 15, len(idx))
marquer(idx, "variance_elevee")

# ------------------------------------------------------------------
# 5. CONTRAINTES PHYSIQUES
# ------------------------------------------------------------------
# Un débit ne peut être ni négatif, ni supérieur à la capacité du lien
debit = np.clip(debit, 0, CAPACITE_MBPS)
df["debit_mbps"] = debit.round(2)

# ------------------------------------------------------------------
# 6. VALEURS MANQUANTES (comme Zabbix quand un équipement est injoignable)
# ------------------------------------------------------------------
# 20 mesures perdues au hasard...
manquantes = rng.choice(len(df), size=20, replace=False)
df.loc[manquantes, "debit_mbps"] = np.nan
# ... et un trou d'une heure (jour 7, 03h00 à 04h00) : équipement injoignable
trou = indices(jour=7, heure_debut=3, duree_min=60)
df.loc[trou, "debit_mbps"] = np.nan
# Ces trous ne sont PAS des anomalies réseau : ils restent étiquetés 0

# ------------------------------------------------------------------
# 7. SAUVEGARDE
# ------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
df = df[["timestamp", "debit_mbps", "anomalie", "type_anomalie"]]
df.to_csv("data/reseau_debit.csv", index=False)

print(f"Fichier créé : data/reseau_debit.csv ({len(df)} lignes)")
print("Valeurs manquantes :", int(df["debit_mbps"].isna().sum()))
print("\nAnomalies simulées :")
print(df[df["anomalie"] == 1].groupby("type_anomalie").size().rename("nb_mesures"))

# ------------------------------------------------------------------
# 8. GRAPHIQUE DE CONTRÔLE (facultatif)
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")           # permet de sauvegarder sans ouvrir de fenêtre
    import matplotlib.pyplot as plt

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["debit_mbps"], lw=0.6, label="Débit")
    anorm = df[df["anomalie"] == 1]
    ax.scatter(anorm["timestamp"], anorm["debit_mbps"], s=6, color="red", label="Anomalie (vérité terrain)")
    ax.set_ylabel("Mbps")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/01_donnees_v1.png", dpi=120)
    print("Figure créée : figures/01_donnees_v1.png")
except ImportError:
    print("matplotlib absent : graphique ignoré")
