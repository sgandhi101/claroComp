import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data into a pandas dataframe
data = pd.read_csv('patient_data.csv')

# Extract the length of stay column
los = data['LOS'].values

# Normalize the data to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
los = scaler.fit_transform(los.reshape(-1, 1))

# Convert the data into a tensor
los = torch.from_numpy(los).float()

# Split the data into a training set and a test set
training_data = los[:int(0.8 * len(los))]
test_data = los[int(0.8 * len(los)):]


# Convert the data into a 3D array to feed into the LSTM model
def create_dataset(data, look_back=1):
    data_X, data_y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        data_X.append(a)
        data_y.append(data[i + look_back, 0])
    return torch.from_numpy(np.array(data_X)).float(), torch.from_numpy(np.array(data_y)).float()


look_back = 1
train_X, train_y = create_dataset(training_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


input_dim = 1
hidden_dim = 10
layer_dim = 1
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

# Define the loss function and optimizer
crit