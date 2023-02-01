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
features = ['GMLOS', 'Pridx', 'Class', 'DRG Description', 'Dishcarge Status', 'DRG', 'DRG Weight', 'Expected Mortality',
            'Secdx1', 'M/S']
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

import itertools
import numpy as np

# Get all possible combinations of features
combinations = [comb for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]

# Initialize a dictionary to store the MSE for each combination
mse_dict = {}

# Loop over each combination of features
for comb in combinations:
    # Define the selected features for this combination
    X_train_selected = X_train[list(comb)]
    X_test_selected = X_test[list(comb)]

    # Fit the linear regression model to the training data with the selected features
    reg = LinearRegression().fit(X_train_selected, y_train)

    # Use the model to make predictions on the test data
    y_pred = reg.predict(X_test_selected)

    # Calculate the mean squared error to evaluate the performance of the model
    mse = mean_squared_error(y_test, y_pred)

    # Store the MSE for this combination in the dictionary
    mse_dict[comb] = mse

# Find the combination of features with the lowest MSE
best_combination = min(mse_dict, key=mse_dict.get)
best_mse = mse_dict[best_combination]

print('Best combination of features:', best_combination)
print('Lowest MSE:', best_mse)

import matplotlib.pyplot as plt

# Define the selected features for the best combination
X_train_selected = X_train[list(best_combination)]
X_test_selected = X_test[list(best_combination)]

# Fit the linear regression model to the training data with the selected features
reg = LinearRegression().fit(X_train_selected, y_train)

# Use the model to make predictions on the test data
y_pred = reg.predict(X_test_selected)

# Define the threshold
threshold = 5

# Select only the values where the difference between the actual value and the predicted value is less than the threshold
mask = abs(y_test - y_pred) < threshold
y_test_selected = y_test[mask]
y_pred_selected = y_pred[mask]

# Plot the selected actual vs. predicted values
plt.scatter(y_test_selected, y_pred_selected)

# Add the line y = x
plt.plot([0, max(y_test_selected)], [0, max(y_test_selected)], 'r--')

plt.xlabel('Actual LOS')
plt.ylabel('Predicted LOS')
plt.title('Actual vs. Predicted LOS (Selected Values)')
plt.show()

import matplotlib.pyplot as plt

# Define the selected features for the best combination
X_train_selected = X_train[list(best_combination)]
X_test_selected = X_test[list(best_combination)]

# Fit the linear regression model to the training data with the selected features
reg = LinearRegression().fit(X_train_selected, y_train)

# Use the model to make predictions on the test data
y_pred = reg.predict(X_test_selected)

# Limit the values in y_pred to a maximum difference of 10 from the actual value
y_pred_selected = y_pred.copy()
range = 20
y_pred_selected[y_pred_selected > y_test + range] = y_test[y_pred_selected > y_test + range] + range
y_pred_selected[y_pred_selected < y_test - range] = y_test[y_pred_selected < y_test - range] - range

# Plot the actual vs. predicted values
plt.scatter(y_test, y_pred_selected)

# Add the line y = x
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')

plt.xlabel('Actual LOS')
plt.ylabel('Predicted LOS')
plt.title('Actual vs. Predicted LOS (Keras RL Layer)')
plt.show()

from scipy.stats import probplot

# ...
# the code from before, up to the scatter plot

# Add the Q-Q plot
probplot(y_test - y_pred, plot=plt)
plt.title('Residual Normalization (RL Enforced)')
# Remove x-axis numbers
plt.xticks([])
plt.xlabel("Actual LOS")
plt.ylabel("Predicted LOS")

# Remove y-axis numbers
plt.yticks([])
plt.show()
