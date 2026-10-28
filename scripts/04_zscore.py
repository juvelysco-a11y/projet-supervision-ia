import pandas as pd

df = pd.read_csv("data/reseau_debit.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["moy"]=df["debit_mbps"].rolling(24).mean()
df["ecart"]=df["debit_mbps"].rolling(24).std()
df["z"]=(df["debit_mbps"]-df["moy"])/df["ecart"]
df["alerte_z"]=(df["z"].abs()>3).astype(int)

print(df["alerte_z"].sum())

vp = ((df["alerte_z"]==1)&(df["anomalie"]==1)).sum()
fp = ((df["alerte_z"]==1)&(df["anomalie"]==0)).sum()
fn = ((df["alerte_z"]==0)&(df["anomalie"]==1)).sum()
print(f"VP : {vp} , FP : {fp} , FN : {fn}")

precision = vp/(vp+fp)
rappel=vp/(vp+fn)
print(f"Précision : {precision:.2f}")
print(f"Rappel : {rappel:.2f}")

print(df[df["anomalie"]==1].groupby("type_anomalie")["alerte_z"].mean())
