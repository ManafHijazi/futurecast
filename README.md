# FutureCast

FutureCast is an application designed to predict stock prices using an LSTM neural network. The application fetches stock data from Alpha Vantage, preprocesses it, trains an LSTM model, and makes predictions on the stock price.

## Features

- Fetches stock data from Alpha Vantage API
- Preprocesses the data using MinMaxScaler
- Trains an LSTM model to predict stock prices
- Displays the fetched data and predicted stock prices in a user-friendly interface
- Shows a loading animation during the process

## Requirements

- Python 3.7+
- Streamlit
- Requests
- Numpy
- Pandas
- Scikit-learn
- TensorFlow
- Joblib
- Dotenv

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/manafhijazi/futurecast.git
   cd futurecast
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory and add your Alpha Vantage API key:
   ```env
   API_KEY=your_alpha_vantage_api_key
   ```

## Usage

1. Activate the virtual environment (if not already activated):

   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501` to view the app.

## File Structure

```plaintext
.
├── app.py                  # Main application file
├── requirements.txt        # Required Python packages
├── .env                    # Environment variables (API key)
├── README.md               # This file
└── venv/                   # Virtual environment directory
```

## Contributing

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -m 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
