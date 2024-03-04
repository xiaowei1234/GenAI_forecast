# Import necessary libraries
import streamlit as st
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Model class (should be the same as the one used for training)
class SimpleRNN(nn.Module):
    # Model definition as before
    # ...

@st.cache(allow_output_mutation=True)
def load_model(line_group_name):
    model = SimpleRNN(input_size=2, hidden_size=10, output_size=2)
    model.load_state_dict(torch.load(f'./model/{line_group_name}.pt'))
    model.eval()
    return model

# Streamlit UI
st.title('Time Series Forecasting')

line_group_name = st.text_input('Enter LINE_GROUP_NAME:')
num_days = st.number_input('Enter the number of days to forecast:', min_value=1, max_value=365, value=1)

if st.button('Forecast'):
    model = load_model(line_group_name)
    
    # User input for the initial time series
    user_input = st.text_area("Enter your time series data in CSV format (DATE, TOTAL_QUANTITY, ON_PROMOTION):")
    input_df = pd.read_csv(pd.compat.StringIO(user_input))
    input_df['DATE'] = pd.to_datetime(input_df['DATE'])
    inputs = input_df[['TOTAL_QUANTITY', 'ON_PROMOTION']].values
    inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1, 2)  # Reshape to (batch, seq, feature)
    
    # Initialize forecast dataframe
    forecast_df = pd.DataFrame(columns=['DATE', 'TOTAL_QUANTITY', 'ON_PROMOTION'])
    
    # Iteratively generate forecasts
    for _ in range(num_days):
        forecast = model(inputs).detach().numpy()
        last_date = input_df['DATE'].iloc[-1] if forecast_df.empty else forecast_df['DATE'].iloc[-1]
        forecast_date = last_date + timedelta(days=1)
        forecast_df = forecast_df.append({'DATE': forecast_date, 'TOTAL_QUANTITY': forecast[0, 0], 'ON_PROMOTION': forecast[0, 1]}, ignore_index=True)
        
        # Update inputs with the new forecast for the next iteration
        new_input = torch.tensor([[forecast[0, 0], forecast[0, 1]]], dtype=torch.float32).view(1, -1, 2)
        inputs = torch.cat((inputs[:, 1:, :], new_input), dim=1)  # Slide window to include new forecast
        
    forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%Y-%m-%d')  # Format dates
    
    # Display results
    st.write(f"Forecasted values for the next {num_days} days:")
    st.dataframe(forecast_df)