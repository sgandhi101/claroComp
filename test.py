import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
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

unique_values = df['DRG Description'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['DRG Description'] = df['DRG Description'].map(value_map)

unique_values = df['Secdx1'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['Secdx1'] = df['Secdx1'].map(value_map)

unique_values = df['M/S'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['M/S'] = df['M/S'].map(value_map)

# Define the features and target
features = ['DRG Description']
target = 'LOS'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=0)

# Fit the linear regression model to the training data
reg = LinearRegression().fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the mean squared error to evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error:', mse)
import matplotlib.pyplot as plt

# Plot the actual vs. predicted values
plt.scatter(y_test, y_pred)

plt.xlabel('Actual LOS')
plt.ylabel('Predicted LOS')
plt.title('Actual vs. Predicted LOS')

plt.show()
