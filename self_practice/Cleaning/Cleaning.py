import pandas as pd

# Use raw string for the file path to avoid invalid escape sequence issues
data = pd.read_csv(r"D:\MCA Collage\Sem-3\Machine Learning\self_practice\Cleaning\RawData.csv")

print(data.columns)
print(len(data.columns))
print(len(data))
print(data.dtypes)
print(data.isnull().values.any())

print("\nTotal empty cells by column :\n", data.isnull().sum(), "\n\n")

print("\n\nNumber of Unique Locations : ", len(data['Location'].unique()))

print("\n\nNumber of Unique Salaries : ", len(data['Salary'].unique()))
print(data['Salary'].unique())

# Cleaning the experience
exp = list(data.Experience)
min_ex = []
max_ex = []

# Populate the min_ex and max_ex lists
for i in range(len(exp)):
    exp[i] = exp[i].replace("yrs", "").strip()
    min_ex.append(int(exp[i].split("-")[0].strip()))
    max_ex.append(int(exp[i].split("-")[1].strip()))

# Move the column assignments outside the loop
data["minimum_exp"] = min_ex
data["maximum_exp"] = max_ex

# Label encoding location and salary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])
data['Salary'] = le.fit_transform(data['Salary'])

# Save the cleaned data to a new CSV file
df = data[['Index', 'Company', 'Location', 'Salary', 'minimum_exp', 'maximum_exp']]
df.to_csv(r"D:\MCA Collage\Sem-3\Machine Learning\self_practice\Cleaning\File5.csv", index=False)

# Read New dataset
data = pd.read_csv(r"D:\MCA Collage\Sem-3\Machine Learning\self_practice\Cleaning\File5.csv")
print(data)
