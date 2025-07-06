import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    Calculate technical indicators for stock
    
    Parameters:
        data: DataFrame, DataFrame containing OHLCV data
        start_date: str, Start date (optional, used for relative performance calculation)
        end_date: str, End date (optional, used for relative performance calculation)
    
    Returns:
        DataFrame: Data with added technical indicators
    """
    # Add date features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # Moving averages
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()
    
    # RSI indicator
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD indicator
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # VWAP indicator
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Bollinger Bands
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # Relative performance to benchmark
    if start_date and end_date:
        benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
        data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100
    
    # ROC indicator
    data['ROC'] = data['Close'].pct_change(periods=1) * 100
    
    # ATR indicator
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # Previous day data
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)
    
    # Remove missing values
    data = data.dropna()
    
    return data

def get_stock_data(ticker, start_date, end_date):
    """
    Get and process data for a single stock
    
    Parameters:
        ticker: Stock symbol
        start_date: Start date
        end_date: End date
    Returns:
        Processed stock data DataFrame
    """
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)  # No proxy
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data, start_date, end_date)
    
    return data

def clean_csv_files(file_path):

    df = pd.read_csv(file_path)
            
    # Delete the second and third rows
    df = df.drop([0, 1]).reset_index(drop=True)
            
    # Rename columns
    df = df.rename(columns={'Price': 'Date'})
            
    # Save the modified file
    df.to_csv(file_path, index=False)
    print("All files processed!")

def main():
    """Main function: Execute data collection and processing workflow"""
    # Stock category list
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # Technology
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # Finance
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # Pharmaceutical
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # Energy
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # Consumer
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # Industrial
    ]

    # Set parameters
    START_DATE = '2020-01-01'
    END_DATE = '2025-04-05'
    NUM_FEATURES_TO_KEEP = 9
    
    # Create data folder
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # Get and save all stock data
    print("Starting to download and process stock data...")
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            stock_data = get_stock_data(ticker, START_DATE, END_DATE)
            stock_data.to_csv(f'{data_folder}/{ticker}.csv')
            clean_csv_files(f'{data_folder}/{ticker}.csv')
            print(f"{ticker} processing completed")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()