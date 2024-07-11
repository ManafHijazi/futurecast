import os
import time
import torch
import joblib
import logging
import numpy as np
import pandas as pd
import yfinance as yf
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

# Function to fetch data from Yahoo Finance
def fetch_data_yahoo(symbol='AAPL'):
    if not symbol:
        logging.error("Stock symbol is not provided.")
        st.error("Stock symbol is not provided.")
        return None
    logging.info(f"Fetching data for symbol: {symbol}")
    df = yf.download(symbol, period='1y')
    logging.info(f"API response data:\n{df.head()}\n{df.tail()}\nData Length: {len(df)}")
    if df.empty:
        logging.error(f"Failed to fetch data for symbol: {symbol}")
        st.error(f"Failed to fetch data from Yahoo Finance.")
        return None
    return df

# Function to add technical indicators
def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
    df = df.dropna()
    logging.info(f"Data after adding indicators:\n{df.head()}\n{df.tail()}")
    return df

# Function to calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate MACD
def calculate_macd(series, span1=12, span2=26, span3=9):
    ema1 = series.ewm(span=span1, adjust=False).mean()
    ema2 = series.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=span3, adjust=False).mean()
    return macd, signal

# Function to preprocess data
def preprocess_data(df):
    logging.info("Preprocessing data")
    df = add_indicators(df)
    logging.info(f"Data after adding indicators:\n{df.head()}\n{df.tail()}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal_Line']])
    logging.info(f"Scaled data shape: {scaled_data.shape}")

    # Dynamically set the look-back period based on available data
    look_back = min(252, len(scaled_data) - 1)
    logging.info(f"Using look-back period: {look_back}")

    def create_dataset(data, look_back):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, look_back)

    logging.info(f"Created dataset X shape: {X.shape}, Y shape: {Y.shape}")

    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Not enough data to create a dataset. Try a different stock symbol or ensure there's sufficient data.")

    return X, Y, scaler

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
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
    hidden_size = 100
    num_layers = 3
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
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
def load_model_and_scaler(input_size):
    logging.info("Loading the model and scaler")
    hidden_size = 100
    num_layers = 3
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
        df = fetch_data_yahoo(symbol)
    if df is not None:
        st.write('Fetched data:')
        st.dataframe(df)
        logging.info("Data fetching step completed")

        # Preprocessing data
        logging.info("Starting data preprocessing step")
        try:
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
        except ValueError as e:
            st.error(str(e))

# Load model and scaler if they exist
if os.path.exists('stock_price_model.pth') and os.path.exists('scaler.pkl'):
    input_size = 8  # Adjusted to match the number of features
    model, scaler = load_model_and_scaler(input_size)
else:
    model, scaler = None, None

# Prediction
if model:
    if st.button('Predict'):
        logging.info("Starting prediction step")
        with st.spinner('Predicting...'):
            df = fetch_data_yahoo(symbol)
            if df is not None:
                df = add_indicators(df)  # Ensure indicators are present in the latest data
                latest_data = df[['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal_Line']].values[-252:]
                latest_data = latest_data[::-1]  # Reverse to have the latest data first

                input_scaled = scaler.transform(latest_data)  # Ensure the correct shape
                input_reshaped = torch.tensor(input_scaled, dtype=torch.float32).view(1, -1, input_scaled.shape[1])
                model.eval()
                with torch.no_grad():
                    prediction = model(input_reshaped)
                prediction_unscaled = scaler.inverse_transform(np.tile(prediction.numpy(), (1, input_scaled.shape[1])))[:, 0]
                predicted_price = prediction_unscaled[0]
                previous_close_price = df['Close'].iloc[-2]
                if predicted_price > previous_close_price:
                    st.markdown(f'**<span style="color:green">Predicted Stock Price: {predicted_price}</span>**', unsafe_allow_html=True)
                else:
                    st.markdown(f'**<span style="color:red">Predicted Stock Price: {predicted_price}</span>**', unsafe_allow_html=True)

                logging.info("Prediction step completed")

                # Display input features for confirmation
                st.write('Latest Close Prices (used as input features):')
                st.write(pd.DataFrame(latest_data, columns=['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal_Line']))
