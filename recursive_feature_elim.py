from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
X.drop(X.columns.difference(['GMLOS', 'Expected Mortality', 'Class', 'DRG', 'DRG Weight', 'Pridx']), 1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Select the number of features to keep
num_features = 1

# Create a random forest model
model = RandomForestRegressor()

# Create an RFE model and select the number of features
rfe = RFE(model)
rfe = rfe.fit(X_train, y_train)

# Print the features selected by RFE
print("Features selected by RFE:", X_train.columns[rfe.support_])

# Make predictions using the RFE-selected features
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_test_rfe = X_test[X_test.columns[rfe.support_]]
model_rfe = LinearRegression()
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)

# Evaluate the model using mean squared error
mse_rfe = mean_squared_error(y_test, y_pred_rfe)
r2_rfe = r2_score(y_test, y_pred_rfe)
print("Mean Squared Error with RFE features:", mse_rfe)
print("R-Squared with RFE features:", r2_rfe)
