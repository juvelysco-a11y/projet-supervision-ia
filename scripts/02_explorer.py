import pandas as pd

df = pd.read_csv("data/reseau_debit.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(df.head())
print(df.info())
print(df.describe())

print(df.isna().sum())
df["heure"] = df["timestamp"].dt.hour
print(df.groupby("heure")["debit_mbps"].mean())