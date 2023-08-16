import json
import pandas as pd

jsonfile = open('credit_default.json')
data = json.load(jsonfile)

df = pd.DataFrame(data, columns=['age', 'sex', 'default'])

# Umur
mean_age = int(df['age'].mean())
print("Rata-rata umur pelanggan: " + str(mean_age) + " tahun")

# Jenis kelamin
customer_by_sex = df['sex'].value_counts(normalize=True) * 100
customer_by_sex = customer_by_sex.rename(index={'male': 'Laki-laki', 'female': 'Perempuan'})
print("Proporsi pelanggan berdasarkan jenis kelamin:")
for index, value in customer_by_sex.items():
    print(f"{index}: {value:.2f}%")

# Default
default_credit = df[df['default'] == 1]
default_credit_percent = (len(default_credit) / len(df['default'])) * 100
print("Proporsi pelanggan yang mengalami default: " + str(default_credit_percent) + "%")

jsonfile.close()