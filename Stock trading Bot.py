import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.ensemble import RandomForestClassifier

# Download historical stock data
def get_stock_data(ticker, period="1y", interval="1d"):
    stock = yf.download(ticker, period=period, interval=interval)
    return stock

# Feature engineering for trading strategy
def generate_features(data):
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'])
    data.dropna(inplace=True)
    return data

# Label the data for machine learning
def generate_labels(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    return data

# Train a simple trading model
def train_model(data):
    features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal']
    X = data[features]
    y = data['Target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict buy/sell signals
def predict_trade(model, data):
    features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal']
    data['Prediction'] = model.predict(data[features])
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    stock_data = get_stock_data(ticker)
    stock_data = generate_features(stock_data)
    stock_data = generate_labels(stock_data)
    model = train_model(stock_data)
    stock_data = predict_trade(model, stock_data)
    
    print(stock_data[['Close', 'Prediction']].tail(10))
