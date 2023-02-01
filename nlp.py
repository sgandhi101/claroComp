import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the data into a pandas dataframe
df = pd.read_csv('inpatient_discharges_first_six.csv')

# Split the data into training and testing sets
training_data = df[:int(len(df) * 0.8)]
testing_data = df[int(len(df) * 0.8):]

# Convert the DRG descriptions into numerical features using Tf-Idf
vectorizer = TfidfVectorizer()
training_features = vectorizer.fit_transform(training_data['DRG Description'])
testing_features = vectorizer.transform(testing_data['DRG Description'])

# Train a logistic regression model to predict the length of stay
model = LogisticRegression()
model.fit(training_features, training_data['LOS'])

# Make predictions on the testing data
predictions = model.predict(testing_features)

# Evaluate the model's accuracy
accuracy = model.score(testing_features, testing_data['LOS'])
print("Accuracy:", accuracy)
