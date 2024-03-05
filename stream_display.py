# Import necessary libraries
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from io import StringIO


# Model class (should be the same as the one used for training)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out


@st.cache_resource
def load_model():
    model = SimpleRNN(input_size=2, hidden_size=10, output_size=2)
    model.load_state_dict(torch.load(f'./model/forecast_model.pt'))
    model.eval()
    return model


model = load_model()

# Streamlit UI
st.title('Time Series Forecasting')

num_days = st.number_input('Enter the number of days to forecast:', min_value=1, max_value=365, value=1)
user_input = st.text_area("Enter your time series data in CSV format (TOTAL_QUANTITY, ON_PROMOTION):")

if st.button('Forecast'):
    # User input for the initial time series

    input_df = pd.read_csv(StringIO(user_input), names=['TOTAL_QUANTITY', 'ON_PROMOTION'], header=None)
    # input_df['DATE'] = pd.to_datetime(input_df['DATE'])
    inputs = input_df[['TOTAL_QUANTITY', 'ON_PROMOTION']].values
    inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1, 2)  # Reshape to (batch, seq, feature)
    
    # Initialize forecast dataframe
    forecast_df = pd.DataFrame(columns=['TOTAL_QUANTITY', 'ON_PROMOTION'])
    
    # Iteratively generate forecasts
    for i in range(num_days):
        forecast = model(inputs).detach().numpy()
        # last_date = input_df['DATE'].iloc[-1] if forecast_df.empty else forecast_df['DATE'].iloc[-1]
        # forecast_date = last_date + timedelta(days=1)
        forecast_df.loc[i] = (forecast[0, 0], forecast[0, 1])
        
        # Update inputs with the new forecast for the next iteration
        new_input = torch.tensor([[forecast[0, 0], forecast[0, 1]]], dtype=torch.float32).view(1, -1, 2)
        inputs = torch.cat((inputs[:, 1:, :], new_input), dim=1)  # Slide window to include new forecast
        
    # forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%Y-%m-%d')  # Format dates
    
    # Display results
    st.write(f"Forecasted values for the next {num_days} days:")
    st.dataframe(forecast_df)