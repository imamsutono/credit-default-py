import json
import pandas as pd

jsonfile = open('credit_default.json')
data = json.load(jsonfile)

df = pd.DataFrame(data, columns=['age', 'default'])

default_credit = df[df['default'] == 1]
default_credit_percent = (len(default_credit) / len(df['default'])) * 100
print("Proporsi pelanggan yang mengalami default: " + str(default_credit_percent) + "%")

jsonfile.close()