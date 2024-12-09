 # import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Excel file
play_tennis = pd.read_excel(r"D:\MCA Collage\Sem-3\Machine Learning\self_practice\Play_tennies\PlayTennis.xlsx")
play_tennis.head()

# Encode the categorical variables
number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])

# Define the features and the target variables
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

# Split the data into training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    play_tennis[features], play_tennis[target], test_size=0.33, random_state=54
)

# Create the Naive Bayes model
model = GaussianNB()
model.fit(features_train, target_train)

# Make predictions on the test set
pred = model.predict(features_test)

# Calculate accuracy
accuracy = accuracy_score(target_test, pred)
print(f"Model Accuracy: {accuracy}")

# Fixing the warning by using a DataFrame with the correct feature names for predictions
sample_data = pd.DataFrame([[1, 2, 0, 1], [2, 0, 0, 0], [0, 0, 0, 1]], columns=features)

# Make predictions on new examples
print(model.predict(sample_data))
