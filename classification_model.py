import json
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

jsonfile = open('credit_default.json')
data = json.load(jsonfile)
# df = pd.DataFrame(data, columns=['age', 'avg_saving_balance', 'avg_checking_balance', 'avg_credit_amt', 'avg_duration', 'default'])
df = pd.DataFrame(data)

# Coba perbaikan data dengan mengisi nilai yang null
# numeric_columns = ['age', 'avg_saving_balance', 'avg_checking_balance', 'avg_credit_amt', 'avg_duration', 'default']
# df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# df['job'].dropna(inplace=True)

# Konversi variabel kategorikal menjadi variabel dummy
df = pd.get_dummies(df, columns=['sex', 'job', 'housing', 'purpose'], drop_first=True)

# Pisahkan atribut dan target
X = df.drop('default', axis=1)
y = df['default']

# Bagi dataset menjadi pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Regresi Logistik
logreg = LogisticRegression()

# Latih model pada dataset pelatihan
logreg.fit(X_train, y_train)

# Lakukan prediksi pada dataset pengujian
y_pred = logreg.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
csf_rep = classification_report(y_test, y_pred, zero_division=1)

print("Akurasi:", accuracy)
print("Classification Report:\n", csf_rep)

# Definisikan kombinasi hyperparameter yang akan diuji
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Buat objek Grid Search
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')

# Latih model pada dataset pelatihan dengan Grid Search
grid_search.fit(X_train, y_train)

# Hyperparamter terbaik
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluasi performa model terbaik pada data pengujian
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

warnings.filterwarnings("ignore")

print("Hyperparamater terbaik:", best_params)
print("Akurasi model terbaik:", best_accuracy)