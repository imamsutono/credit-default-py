import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

jsonfile = open('credit_default.json')
data = json.load(jsonfile)
df = pd.DataFrame(data, columns=['age', 'avg_saving_balance', 'avg_checking_balance', 'avg_credit_amt', 'avg_duration', 'default'])

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
csf_rep = classification_report(y_test, y_pred)

print("Akurasi:", accuracy)
print("Classification Report:\n", csf_rep)