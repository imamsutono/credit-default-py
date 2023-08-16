import json
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

jsonfile = open('credit_default.json')
data = json.load(jsonfile)

df = pd.DataFrame(data)

default_0 = df[df['default'] == 0]
default_1 = df[df['default'] == 1]

numeric_attributes = ['age', 'avg_saving_balance', 'avg_checking_balance', 'avg_credit_amt', 'avg_duration']
significant_attribute = []

print("Pengaruh atribut terhadap default:")

for attribute in numeric_attributes:
    result = ttest_ind(default_0[attribute], default_1[attribute])
    p_value = result.pvalue
    print(f"{attribute}: {p_value:.4f}")
    if p_value < 0.05:
        significant_attribute.append(attribute)

# Pisahkan atribut dan target
X = df.drop('default', axis=1)
y = df['default']

# Uji chi-squared untuk atribut kategorikal
categorical_attributes = ['sex', 'job', 'housing', 'purpose']

for attribute in categorical_attributes:
    contingency_table = pd.crosstab(X[attribute], y)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"{attribute}: {p:.4f}")
    if p < 0.05:
        significant_attribute.append(attribute)

print("\nAtribut yang memiliki pengaruh signifikan kepada default dengan confidence 95% adalah:")
print(', '.join(significant_attribute))