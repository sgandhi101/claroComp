import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
y = df['LOS']

# Select the features to use for prediction
features = ['Patient Type', 'Admit Date', 'Discharge Date', 'Expected Mortality', 'GMLOS',
            'Dishcarge Status', 'M/S', 'Class', 'DRG', 'DRG Weight', 'DRG Description',
            'Pridx', 'Secdx1', 'Secdx2', 'Secdx3', 'Secdx4', 'Secdx5', 'Secdx6', 'Secdx7',
            'Secdx8', 'Secdx9', 'Secdx10', 'Secdx11', 'Secdx12', 'Secdx13', 'Secdx14',
            'Secdx15', 'Secdx16', 'Secdx17', 'Secdx18', 'Secdx19', 'Secdx20', 'Secdx21',
            'Secdx22', 'Secdx23', 'Secdx24', 'Secdx25']
X = df[features]
X.drop(X.columns.difference(['GMLOS', 'Expected Mortality', 'DRG', 'DRG Weight', 'Pridx']), 1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the model
nn = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, early_stopping=True, validation_fraction=0.2, alpha=0.01)

# Use grid search to tune the hyperparameters
param_grid = {'alpha': [0.01, 0.1, 1.0],
              'hidden_layer_sizes': [(50, 50), (100, 100), (150, 150)]}
grid = GridSearchCV(nn, param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

# Make predictions on the test data
y_pred = grid.predict(X_test)

# Calculate the Root Mean Squared Error of the predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_test, residuals)
plt.xlabel('Actual LOS (y_test)')
plt.ylabel('Residuals (y_test - y_pred)')
plt.title('Residual Plot')
plt.hlines(y=0, xmin=min(y_test), xmax=max(y_test), color='red', linestyle='dashed')
plt.show()

import seaborn as sns

sns.swarmplot(x=y_test, y=y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Swarm Plot of True Values vs Predictions')
plt.show()

sns.pairplot(df[['GMLOS', 'Expected Mortality', 'DRG', 'DRG Weight', 'Pridx', 'LOS']])
plt.show()

# Calculate the correlation matrix
corr = X_train.corr()

# Create the heatmap
sns.heatmap(corr, annot=True)

# Show the plot
plt.show()
