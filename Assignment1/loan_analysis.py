# Kacper Winder - ITC 3154 Assignment 1

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder = Path(__file__).parent
df = pd.read_csv(folder / 'loan_approval_dataset.csv', sep=';', skipinitialspace=True)

df.columns = df.columns.str.strip()
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

print('First 20 rows:')
print(df.head(20).to_string(index=False))
print('\nRows and columns:', df.shape)
print('\nMissing values:')
print(df.isna().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nLoan results:')
print(df['loan_status'].value_counts())


df['loan_status_numeric'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
numbers = df.select_dtypes(include='number')
corrmatrix = numbers.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corrmatrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Loan dataset correlations')
plt.tight_layout()
plt.savefig(folder / 'heatmap.png')
plt.show()

print('\nCorrelations with approval:')
print(corrmatrix['loan_status_numeric'].drop('loan_status_numeric'))

education = pd.crosstab(df['education'], df['loan_status'])
education['Total'] = education['Approved'] + education['Rejected']
education['Approval percent'] = education['Approved'] / education['Total'] * 100
print('\nEducation and approval:')
print(education.round(2))
