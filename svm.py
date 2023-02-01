from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('inpatient_discharges_first_six.csv')

df['Expected Mortality'] = df['Expected Mortality'].str.strip('%')
# convert to float
df['Expected Mortality'] = df['Expected Mortality'].astype(float)
# divide by 100 to get the actual value
df['Expected Mortality'] = df['Expected Mortality'] / 100

# Create a dictionary to map each categorical value to a numerical value
class_map = {'MCC': 1, 'CC': 2, 'NCC': 3}

# Use the map function to convert the 'Class' column
df['Class'] = df['Class'].map(class_map)

unique_values = df['Pridx'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['Pridx'] = df['Pridx'].map(value_map)

# Extract the target column 'Expected Mortality'
y = df['Dishcarge Status']

# Select the features to use for prediction
features = ['Patient Type', 'Admit Date', 'Discharge Date', 'Expected Mortality', 'GMLOS',
            'Dishcarge Status', 'M/S', 'Class', 'DRG', 'DRG Weight', 'DRG Description',
            'Pridx', 'Secdx1', 'Secdx2', 'Secdx3', 'Secdx4', 'Secdx5', 'Secdx6', 'Secdx7',
            'Secdx8', 'Secdx9', 'Secdx10', 'Secdx11', 'Secdx12', 'Secdx13', 'Secdx14',
            'Secdx15', 'Secdx16', 'Secdx17', 'Secdx18', 'Secdx19', 'Secdx20', 'Secdx21',
            'Secdx22', 'Secdx23', 'Secdx24', 'Secdx25']
X = df[features]
X.drop(X.columns.difference(['DRG','Pridx']), 1, inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the SVR model
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model performance
print('Mean Squared Error:', y_pred)

import matplotlib.pyplot as plt

# Plot the actual target values (y_test) against the predicted target values (y_pred)
plt.scatter(y_test, y_pred)

# Plot the line y=x to show the ideal predictions
plt.plot(y_test, y_test, color='red')

# Set the x and y axis labels
plt.xlabel('True LOS')
plt.ylabel('Predicted LOS')

# Show the plot
plt.show()
