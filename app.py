import os
import time
import joblib
import logging
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LSTM # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set the document title
st.set_page_config(page_title="FutureCast")

# URLs for Lottie animations
loading_animation_url = "https://lottie.host/f6db9d22-0c9f-45bd-8256-19c3155fe914/gyZXBVFG7q.json"

# Load Lottie animation from URL
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to load animation from {url}")
        return None
    return response.json()

# Load the general loading animation
loading_animation = load_lottie_url(loading_animation_url)

# Function to fetch data from Alpha Vantage
def fetch_data(symbol='AAPL', api_key=os.getenv('API_KEY')):
    logging.info(f"Fetching data for symbol: {symbol}")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact'
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data: {response.status_code}")
    data = response.json()
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

# Function to build and train the model
def build_and_train_model(X, Y, progress_callback):
    logging.info("Building and training the model")
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    epochs = 20
    for epoch in range(epochs):
        logging.info(f"Training epoch: {epoch + 1}/{epochs}")
        model.fit(X, Y, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        progress_callback(epoch + 1, epochs)

    return model

# Function to save the model and scaler
def save_model(model, scaler):
    logging.info("Saving the model and scaler")
    model.save('stock_price_model.h5')
    joblib.dump(scaler, 'scaler.pkl')

# Function to load the model and scaler
def load_model_and_scaler():
    logging.info("Loading the model and scaler")
    model = load_model('stock_price_model.h5')
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
if os.path.exists('stock_price_model.h5') and os.path.exists('scaler.pkl'):
    model, scaler = load_model_and_scaler()
else:
    model, scaler = None, None

# Prediction
if model:
    if st.button('Predict'):
        logging.info("Starting prediction step")
        with st.spinner('Predicting...'):
            df = fetch_data(symbol)
            latest_data = df['close'].values[:5]
            latest_data = latest_data[::-1]  # Reverse to have the latest data first

            input_scaled = scaler.transform(latest_data.reshape(1, -1))
            input_reshaped = input_scaled.reshape((1, 5, 1))
            prediction = model.predict(input_reshaped)
            prediction_unscaled = scaler.inverse_transform(prediction)
            st.write(f'Predicted Stock Price: {prediction_unscaled[0][0]}')
        logging.info("Prediction step completed")

        # Display input features for confirmation
        st.write('Latest Close Prices (used as input features):')
        st.write(pd.DataFrame(latest_data.reshape(1, -1), columns=[f'Feature {i + 1}' for i in range(len(latest_data))]))

# Add the general loading animation
if loading_animation:
    st_lottie(loading_animation, height=450, key="loading")
