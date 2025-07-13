import os
import time
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings('ignore')

import os
# 从环境变量获取API密钥
API_KEY = os.getenv('TWELVE_DATA_API_KEY')
if not API_KEY:
    raise ValueError('请设置TWELVE_DATA_API_KEY环境变量')

def get_daily_kline(symbol, interval='1day', max_retries=3):
    """
    从 Twelve Data 获取指定股票的日K线数据，包含重试机制
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 5000,
        "apikey": API_KEY
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 检查HTTP错误状态码
            data = response.json()
            
            # 检查API返回的错误
            if 'error' in data:
                raise Exception(f"API Error: {data['error']}")
            
            return data
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"获取 {symbol} 失败，已达到最大重试次数: {str(e)}")
                return {}
            
            # 指数退避重试
            sleep_time = 2 ** retries
            print(f"获取 {symbol} 失败，将在 {sleep_time} 秒后重试 (重试 {retries}/{max_retries}): {str(e)}")
            time.sleep(sleep_time)

    if not data or 'values' not in data:
        print(f"获取 {symbol} 失败：{data.get('message', '无返回数据')}")
        return pd.DataFrame()

    df = pd.DataFrame(data['values'])
    df.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    df = df.astype(float)

    return df

def compute_indicators(df_stock, begin=None, finish=None):
    """
    计算股票的技术指标

    参数：
        df_stock: DataFrame，包含股票历史行情数据（必须包含 'Open', 'High', 'Low', 'Close', 'Volume' 列）
        begin: 开始时间（可选）
        finish: 结束时间（可选）

    返回：
        df_stock: 添加了多种技术指标后的DataFrame
    """

    # 确保索引为时间格式
    if not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index)

    # 如果数据为空或缺少收盘价列则返回空
    if df_stock.empty or 'Close' not in df_stock.columns:
        print("数据为空或缺少 Close 列，跳过该股票。")
        return pd.DataFrame()

    # 异常值处理：涨跌幅超过 ±30% 视为异常，置为 NaN
    change = df_stock['Close'].pct_change()
    df_stock.loc[change.abs() > 0.3, ['Open', 'High', 'Low', 'Close', 'Volume']] = np.nan

    # 用前向和后向填充法处理缺失值
    df_stock.fillna(method='ffill', inplace=True)
    df_stock.fillna(method='bfill', inplace=True)

    # 添加时间特征
    df_stock['Year'] = df_stock.index.year
    df_stock['Month'] = df_stock.index.month
    df_stock['Day'] = df_stock.index.day

    # 简单移动平均线（SMA）
    df_stock['MA5'] = df_stock['Close'].shift(1).rolling(window=5).mean()
    df_stock['MA10'] = df_stock['Close'].shift(1).rolling(window=10).mean()
    df_stock['MA20'] = df_stock['Close'].shift(1).rolling(window=20).mean()
    df_stock['MA50'] = df_stock['Close'].shift(1).rolling(window=50).mean()

    # 前一日价格数据
    df_stock['Close_yes'] = df_stock['Close'].shift(1)
    df_stock['Open_yes'] = df_stock['Open'].shift(1)
    df_stock['High_yes'] = df_stock['High'].shift(1)
    df_stock['Low_yes'] = df_stock['Low'].shift(1)

    # 相对表现指标
    df_stock['Relative_Performance'] = df_stock['Close'] / df_stock['Close'].shift(1) - 1

    # RSI 相对强弱指标
    price_change = df_stock['Close'].diff()
    rise = price_change.clip(lower=0)  # 上涨部分
    drop = -price_change.clip(upper=0)  # 下跌部分
    mean_rise = rise.rolling(window=14).mean()
    mean_drop = drop.rolling(window=14).mean()
    rs_value = mean_rise / mean_drop
    df_stock['RSI'] = 100 - (100 / (1 + rs_value))

    # MACD 指标：快线 - 慢线
    ema_short = df_stock['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df_stock['Close'].ewm(span=26, adjust=False).mean()
    df_stock['MACD'] = ema_short - ema_long

    # VWAP 成交量加权平均价
    df_stock['VWAP'] = (df_stock['Close'] * df_stock['Volume']).cumsum() / df_stock['Volume'].cumsum()

    # 布林带指标
    window_size = 20
    df_stock['SMA'] = df_stock['Close'].rolling(window=window_size).mean()
    df_stock['Std_dev'] = df_stock['Close'].rolling(window=window_size).std()
    df_stock['Upper_band'] = df_stock['SMA'] + 2 * df_stock['Std_dev']  # 上轨
    df_stock['Lower_band'] = df_stock['SMA'] - 2 * df_stock['Std_dev']  # 下轨

    # OBV 能量潮指标（On-Balance Volume）
    obv = [0]
    for i in range(1, len(df_stock)):
        if df_stock['Close'].iloc[i] > df_stock['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df_stock['Volume'].iloc[i])
        elif df_stock['Close'].iloc[i] < df_stock['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df_stock['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df_stock['OBV'] = obv

    # ADX 平均趋向指标 +DI, -DI 和 ATR（真实波幅范围）
    high = df_stock['High']
    low = df_stock['Low']
    close = df_stock['Close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # 正向/负向趋向指标（+DI 和 -DI）
    df_stock['+DI'] = 100 * (plus_dm.rolling(window=14).mean() / atr)
    df_stock['-DI'] = 100 * (minus_dm.rolling(window=14).mean() / atr)

    # ADX 趋势强度指标
    df_stock['ADX'] = (abs(df_stock['+DI'] - df_stock['-DI']) / (df_stock['+DI'] + df_stock['-DI'])).rolling(window=14).mean() * 100

    # ATR 平均真实波幅
    df_stock['ATR'] = atr

    # 添加前一日收盘价
    df_stock['Prev_Close'] = df_stock['Close'].shift(1)

    # 删除缺失值
    df_stock = df_stock.dropna()

    return df_stock

def fetch_stock_history(stock_symbol, date_from, date_to):
    df_raw = get_daily_kline(stock_symbol)
    if df_raw.empty:
        return pd.DataFrame()

    df_raw = df_raw[(df_raw.index >= date_from) & (df_raw.index <= date_to)]
    df_final = compute_indicators(df_raw, date_from, date_to)
    return df_final

def process_all():
    symbols = [
        'MSFT', 'AAPL', 'UBER', 'IBM', 'NVDA',
        'JPM', 'BAC', 'V', 'MS', 'MA',
        'AMZN', 'MCD', 'NIKE', 'TSLA', 'SBUX',
        'META', 'NFLX', 'TMUS', 'DIS', 'T',
        'LLY', 'TMO', 'MRK', 'ABBV', 'GILD',
        'WM', 'DE', 'BA', 'GE', 'HON',
    ]
    from_date = '2020-01-01'
    to_date = '2024-01-01'
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)

    print("开始下载和处理股票数据...")
    for i, code in enumerate(symbols):
        try:
            print(f"正在处理：{code}")
            df_result = fetch_stock_history(code, from_date, to_date)
            if df_result.empty:
                print(f"{code} 数据为空或处理失败")
            else:
                df_result.to_csv(f'{folder_path}/{code}.csv', index=True, index_label='Date')
                print(f"{code} 处理完成")
        except Exception as err:
            print(f"处理 {code} 时出错: {str(err)}")

        #  控流处理：避免超过 TwelveData 免费版每分钟 8 次限制
        if (i + 1) % 7 == 0:  # 每处理 7 个，停 65 秒
            print("达到 API 限制，休息 65 秒防止触发 rate limit...")
            time.sleep(65)
        else:
            time.sleep(8)

if __name__ == "__main__":
    process_all()