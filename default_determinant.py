import json
import pandas as pd

jsonfile = open('credit_default.json')
data = json.load(jsonfile)

df = pd.DataFrame(data)

# Menghitung korelasi antara atribut numerik dengan atribut target
correlation_matrix = df.corr(numeric_only=True)
default_correlation = correlation_matrix['default'].sort_values(ascending=False)

print("Korelasi atribut numeric dengan default:")
print(default_correlation)
print(f"\nAtribut {default_correlation.index[1]} mempunyai pengaruh paling kuat terhadap default")
