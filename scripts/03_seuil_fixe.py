import pandas as pd

df = pd.read_csv("data/reseau_debit.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

SEUIL = 60

df["alerte_seuil"] = (df["debit_mbps"] > SEUIL).astype(int)
print(df["alerte_seuil"].sum())

vp = ((df["alerte_seuil"]==1)&(df["anomalie"]==1)).sum()
fp = ((df["alerte_seuil"]==1)&(df["anomalie"]==0)).sum()
fn = ((df["alerte_seuil"]==0)&(df["anomalie"]==1)).sum()
print(f"VP : {vp} , FP : {fp} , FN : {fn}")

precision = vp/(vp+fp)
rappel=vp/(vp+fn)
print(f"Précision : {precision:.2f}")
print(f"Rappel : {rappel:.2f}")

print(df[df["anomalie"]==1].groupby("type_anomalie")["alerte_seuil"].mean())
