import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Load the data into a pandas dataframe
df = pd.read_csv("inpatient_discharges_first_six.csv")

# Split the data into training and test sets
train = df[:int(0.8*len(df))]
test = df[int(0.8*len(df)):]

# Fit the exponential smoothing model on the training data
model = SimpleExpSmoothing(train["LOS"]).fit()

# Use the model to make predictions on the test data
predictions = model.predict(start=test.index[0], end=test.index[-1])

# Evaluate the model's performance
print("Mean Absolute Error: ", mean_absolute_error(test["LOS"], predictions))

import matplotlib.pyplot as plt

# Plot the actual values of GMLOS on the test data
plt.plot(test.index, test["LOS"], label="Actual")

# Plot the predicted values of GMLOS
plt.plot(test.index, predictions, label="Predicted", linestyle='--')

# Add a legend to the plot
plt.legend()

# Label the x and y axes
plt.xlabel("Time")
plt.ylabel("GMLOS")

# Show the plot
plt.show()
