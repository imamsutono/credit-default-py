import json
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

jsonfile = open('credit_default.json')
data = json.load(jsonfile)

df = pd.DataFrame(data)

# Konversi variabel kategorikal menjadi variabel dummy
df = pd.get_dummies(df, columns=['sex', 'housing', 'purpose'], drop_first=True)

# Mengisi nilai yang hilang pada atribut numerik dengan rata-rata
numeric_columns = ['age', 'job', 'avg_saving_balance', 'avg_checking_balance', 'avg_credit_amt', 'avg_duration']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# df.fillna(df.mean(), inplace=True)

# Pisahkan atribut dan target
X = df.drop('default', axis=1)
y = df['default']

# Pemeriksaan apakah masih ada nilai NaN setelah pengisian dengan mean
if df.isnull().values.any():
    print("Masih terdapat nilai NaN dalam dataframe setelah pengisian dengan mean.")

# Lakukan uji chi-squared untuk atribut kategorikal
for column in X.columns:
    contingency_table = pd.crosstab(X[column], y)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    if p < 0.05:  # Tingkat kepercayaan 95%
        print(f"{column}: Pengaruh signifikan (p-value: {p:.4f})")

# Lakukan analisis regresi logistik untuk atribut numerik
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pemeriksaan apakah masih ada nilai NaN setelah pemisahan dataset
if X_train.isnull().values.any() or X_test.isnull().values.any():
    print("Masih terdapat nilai NaN dalam dataset setelah pemisahan dataset.")

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

coefficients = logreg.coef_[0]

print("\n-- Atribut yang mempengaruhi default--\n")

for i, column in enumerate(X.columns):
    print(f"{column}: Coefficient: {coefficients[i]:.4f}")

# Menghitung korelasi antara atribut numerik dengan atribut target
# correlation_matrix = df.corr(numeric_only=True)
# default_correlation = correlation_matrix['default'].sort_values(ascending=False)
# print("Korelasi atribut dengan default:")
# print(default_correlation)
# print(f"\nAtribut {default_correlation.index[1]} mempunyai pengaruh paling kuat terhadap default")
