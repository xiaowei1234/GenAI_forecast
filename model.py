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
data.sort_values(by=['LINE_GROUP_NAME', 'DATE'], inplace=True)

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

def train_model(dataframe, line_group_name):
    # Prepare the data for training
    group_data = dataframe[dataframe['LINE_GROUP_NAME'] == line_group_name]
    inputs = group_data[['TOTAL_QUANTITY', 'ON_PROMOTION']].values
    inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1, 2)  # Reshape to (batch, seq, feature)

    # Model parameters
    input_size = 2  # TOTAL_QUANTITY, ON_PROMOTION
    hidden_size = 10  # Can be adjusted
    output_size = 2  # We forecast TOTAL_QUANTITY and ON_PROMOTION
    model = SimpleRNN(input_size, hidden_size, output_size)

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
    
    return model

# Train and save models for each LINE_GROUP_NAME
models = {}
for name in data['LINE_GROUP_NAME'].unique():
    model = train_model(data, name)
    models[name] = model
    torch.save(model.state_dict(), f'./model/{name}.pt')

print("Training completed and models saved.")