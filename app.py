import os
import time
import torch
import joblib
import logging
import requests
import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set the document title
st.set_page_config(page_title="FutureCast")

# Load Lottie animation from URL
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to load animation from {url}")
        return None
    return response.json()

# Function to fetch data from Alpha Vantage
def fetch_data(symbol='AAPL', api_key=os.getenv('API_KEY')):
    logging.info(f"Fetching data for symbol: {symbol}")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact'
    response = requests.get(url)
    logging.info(f"API response status code: {response.status_code}")
    if response.status_code != 200:
        logging.error(f"Failed to fetch data: {response.status_code}")
        st.error(f"Failed to fetch data from Alpha Vantage. Status code: {response.status_code}")
        return None
    data = response.json()
    logging.info(f"API response data: {data}")
    if 'Time Series (Daily)' not in data:
        logging.error(f"Key 'Time Series (Daily)' not found in the response. Full response: {data}")
        st.error("Failed to fetch the expected data. Please check the API limits or try again later.")
        return None
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    return df

# Function to preprocess data
def preprocess_data(df):
    logging.info("Preprocessing data")
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])

    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(scaled_data, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, Y, scaler

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to build and train the model
def build_and_train_model(X, Y, progress_callback):
    logging.info("Building and training the model")
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    input_size = X.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        logging.info(f"Training epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}")
        progress_callback(epoch + 1, epochs)

    return model

# Function to save the model and scaler
def save_model(model, scaler):
    logging.info("Saving the model and scaler")
    torch.save(model.state_dict(), 'stock_price_model.pth')
    joblib.dump(scaler, 'scaler.pkl')

# Function to load the model and scaler
def load_model_and_scaler():
    logging.info("Loading the model and scaler")
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('stock_price_model.pth'))
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Streamlit app
st.title('FutureCast')

# Input for stock symbol
symbol = st.text_input('Enter Stock Symbol:', value='')

if st.button('Get Data and Train Model'):
    # Fetching data
    st.write('Fetching data...')
    logging.info("Starting data fetching step")
    with st.spinner('Fetching data...'):
        df = fetch_data(symbol)
    if df is not None:
        st.write('Fetched data:')
        st.dataframe(df)
        logging.info("Data fetching step completed")

        # Preprocessing data
        logging.info("Starting data preprocessing step")
        with st.spinner('Preprocessing data...'):
            X, Y, scaler = preprocess_data(df)
        logging.info("Data preprocessing step completed")

        # Training model
        logging.info("Starting model training step")
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def update_progress(epoch, total_epochs):
            progress_bar.progress(epoch / total_epochs)
            status_text.text(f'Training: Epoch {epoch}/{total_epochs}')
            logging.info(f"Training progress: {epoch}/{total_epochs}")

        with st.spinner('Training model...'):
            model = build_and_train_model(X, Y, update_progress)

        save_model(model, scaler)
        end_time = time.time()
        st.success(f'Model trained in {end_time - start_time:.2f} seconds.')
        logging.info("Model training step completed")

# Load model and scaler if they exist
if os.path.exists('stock_price_model.pth') and os.path.exists('scaler.pkl'):
    model, scaler = load_model_and_scaler()
else:
    model, scaler = None, None

# Prediction
if model:
    if st.button('Predict'):
        logging.info("Starting prediction step")
        with st.spinner('Predicting...'):
            df = fetch_data(symbol)
            if df is not None:
                latest_data = df['close'].values[:5]
                latest_data = latest_data[::-1]  # Reverse to have the latest data first

                input_scaled = scaler.transform(latest_data.reshape(-1, 1))  # Ensure the correct shape
                input_reshaped = torch.tensor(input_scaled, dtype=torch.float32).view(1, -1, 1)
                model.eval()
                with torch.no_grad():
                    prediction = model(input_reshaped)
                prediction_unscaled = scaler.inverse_transform(prediction.numpy())
                st.write(f'Predicted Stock Price: {prediction_unscaled[0][0]}')
                logging.info("Prediction step completed")

                # Display input features for confirmation
                st.write('Latest Close Prices (used as input features):')
                st.write(pd.DataFrame(latest_data.reshape(1, -1), columns=[f'Feature {i + 1}' for i in range(len(latest_data))]))
