import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/reseau_debit.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["heure"] = df["timestamp"].dt.hour

d=df.dropna(subset=["debit_mbps"]).copy()
print("Lignes gardées :",len(d),"sur",len(df))

X=d[["debit_mbps","heure"]]
modele=IsolationForest(contamination=0.035,random_state=42)
modele.fit(X)
d["pred"]=modele.predict(X)

d["alerte_ia"] = (d["pred"]==-1).astype(int)
print("Alertes :", d["alerte_ia"].sum())

vp = ((d["alerte_ia"]==1)&(d["anomalie"]==1)).sum()
fp = ((d["alerte_ia"]==1)&(d["anomalie"]==0)).sum()
fn = ((d["alerte_ia"]==0)&(d["anomalie"]==1)).sum()
print(f"VP : {vp} , FP : {fp} , FN : {fn}")

precision = vp/(vp+fp)
rappel=vp/(vp+fn)
print(f"Précision : {precision:.2f}")
print(f"Rappel : {rappel:.2f}")

print(pd.crosstab(d["anomalie"],d["alerte_ia"]))

print(d[d["anomalie"]==1].groupby("type_anomalie")["alerte_ia"].mean())
