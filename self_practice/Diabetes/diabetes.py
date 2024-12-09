import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv(r'D:\MCA Collage\Sem-3\Machine Learning\self_practice\Diabetes\diabetes.csv', encoding='latin-1')
data

sns.heatmap(data.isnull())

correlation = data.corr()
print(correlation)

X = data.drop("Outcome", axis = 1)
Y = data["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size = 0.2,random_state = 54)
X_train

# model fit
model = LogisticRegression(C=1)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
predictions


accuracy = accuracy_score(predictions, Y_test)
print(accuracy)
