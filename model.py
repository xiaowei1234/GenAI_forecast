# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import os
from datetime import datetime
from collections import defaultdict

# Read the CSV file
data = pd.read_csv('./data/forecast_data.csv')

# Convert DATE column to datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Sort data by LINE_GROUP_NAME and DATE to ensure time series continuity
data = data.set_index(['LINE_GROUP_NAME', 'DATE']).sort_index()


# Simple RNN model definition
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out


def train_model(model, df):
    # Prepare the data for training
    ts_df = df.asfreq('D', fill_value=0.0)
    inputs = ts_df[['TOTAL_QUANTITY', 'ON_PROMOTION']].values
    inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1, 2)  # Reshape to (batch, seq, feature)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):  # Number of epochs can be adjusted
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs[:, -1, :])
        loss.backward()
        optimizer.step()
    


# Model parameters
input_size = 2  # TOTAL_QUANTITY, ON_PROMOTION
hidden_size = 10  # Can be adjusted
output_size = 2  # We forecast TOTAL_QUANTITY and ON_PROMOTION
model = SimpleRNN(input_size, hidden_size, output_size)

# Train and save models for each LINE_GROUP_NAME
for name in data.index.get_level_values(0).unique():
    train_df = data.loc[name]
    print(name)
    if train_df.shape[0] < 2:
        continue
    train_model(model, train_df)

torch.save(model.state_dict(), f'./model/forecast_model.pt')

print("Training completed and models saved.")
