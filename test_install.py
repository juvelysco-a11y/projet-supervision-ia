import pandas as pd
import numpy as np
import sklearn
import statsmodels
import matplotlib
from sklearn.ensemble import IsolationForest

print("pandas       :", pd.__version__)
print("numpy        :", np.__version__)
print("scikit-learn :", sklearn.__version__)
print("statsmodels  :", statsmodels.__version__)
print("matplotlib   :", matplotlib.__version__)

#mini test d'isolation forest : 100 points normauis + 1 point abberant
donnees = np.append(np.random.normal(50, 5, 100), 95).reshape(-1, 1)
modele = IsolationForest(contamination=0.02, random_state=42).fit(donnees)
resultat = modele.predict(donnees)
print("Anomalies détectées :", (resultat == -1).sum())
print("Installation OK")