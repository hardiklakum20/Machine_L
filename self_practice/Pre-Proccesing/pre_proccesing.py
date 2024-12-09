import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Load dataset
dataset = pd.read_csv('D:/MCA Collage/Sem-3/Machine Learning/self_practice/Pre-Proccesing/pre_proccesing.csv')
X = dataset.iloc[:, :-1].values  # Takes all rows of all columns except the last column
Y = dataset.iloc[:, -1].values   # Takes all rows of the last column

# Print column names and dataset info
print(dataset.columns)
dataset.info()
print(dataset.head())
print(dataset.tail())

# Row and column count
print("Dataset shape:", dataset.shape)

# Count missing values
print("Missing values (sorted):\n", dataset.isnull().sum().sort_values(ascending=False))

# Removing an insufficient column
dataset_new = dataset.drop(['Age'], axis=1)
print(dataset_new.describe())

# Rename columns
dataset.rename(columns={'Country': 'Countries', 'Age': 'age', 'Salary': 'Sal', 'Purchased': 'Purchased'}, inplace=True)
print(dataset)

# Print rows with missing values
print("Rows with missing values:\n", dataset[dataset.isnull().any(axis=1)].head())

# Remove missing value rows
ds_new = dataset.dropna()
print("New dataset shape (after dropping rows with missing values):", ds_new.shape)
print("Missing values after dropping rows:", ds_new.isnull().sum())

# Check and convert data types
print("Data types before conversion:\n", ds_new.dtypes)
ds_new['age'] = ds_new['age'].astype('int64', errors='ignore')  # Avoids errors if 'age' is non-numeric
print("Data types after conversion:\n", ds_new.dtypes)

# Impute missing values using mean, median, and most frequent strategies
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("Data after imputation:\n", X)
