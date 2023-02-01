import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the data into a pandas dataframe
df = pd.read_csv("inpatient_discharges_first_six.csv")

# Convert the admit and discharge dates to datetime objects and extract the length of stay
df['Admit Date'] = pd.to_datetime(df['Admit Date'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['LOS'] = (df['Discharge Date'] - df['Admit Date']).dt.total_seconds() / (24 * 60 * 60)

# Create a new dataframe with the length of stay as the target variable and the admit date as the index
ts_df = df[['Admit Date', 'LOS']].set_index('Admit Date')

# Plot the length of stay over time
plt.plot(ts_df)
plt.xlabel('Admit Date')
plt.ylabel('Length of Stay')
plt.title('Length of Stay over Time')
plt.show()

# Split the data into training and test sets
train_data = ts_df[:-50]
test_data = ts_df[-50:]

# Fit the ARIMA model
model = ARIMA(train_data, order=(1, 0, 1))
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1], dynamic=False)

# Calculate the mean squared error of the predictions
mse = np.mean((predictions - test_data['LOS'])**2)
print('Mean Squared Error:', mse)

# Plot the actual vs. predicted length of stay
plt.plot(test_data, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Admit Date')
plt.ylabel('Length of Stay')
plt.title('Actual vs. Predicted Length of Stay')
plt.legend()
plt.show()
