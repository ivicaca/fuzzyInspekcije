from fuzzy_risk_model_single import FuzzyRiskModel
import pandas as pd

# Inicijalizuj model sa putanjom do matrice odlučivanja
model = FuzzyRiskModel("matrica_pregledani.csv")

# Učitaj fajl sa novim subjektima
ulazni_fajl = "SubjektiTestPregledani.csv"
df = pd.read_csv(ulazni_fajl)

# Evaluacija svakog reda pomoću metode evaluate_basic()
rezultati = []
for _, red in df.iterrows():
    rezultat = model.evaluate_basic(red)
    rezultati.append(rezultat)

# Upis rezultata
df['Procena_KSR'] = rezultati

# Snimi novi CSV sa rezultatima
izlazni_fajl = ulazni_fajl.replace(".csv", "_evaluated.csv")
df.to_csv(izlazni_fajl, index=False)
print(f"✅ Evaluacija završena. Rezultati su sačuvani u: {izlazni_fajl}")
print(df)
