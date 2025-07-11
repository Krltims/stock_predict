import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

def compute_indicators(df_stock, begin=None, finish=None):
    """
    计算股票的技术指标
    """
    #处理异常值
    change = df_stock['Close'].pct_change()  # 计算涨跌幅
    df_stock.loc[change.abs() > 0.3, ['Open', 'High', 'Low', 'Close', 'Volume']] = np.nan
    df_stock.fillna(method='ffill', inplace=True)
    df_stock.fillna(method='bfill', inplace=True)

    # 日期特征
    df_stock['Year'] = df_stock.index.year
    df_stock['Month'] = df_stock.index.month
    df_stock['Day'] = df_stock.index.day

    # 中期移动平均
    df_stock['MA10'] = df_stock['Close'].shift(1).rolling(window=10).mean()
    df_stock['MA50'] = df_stock['Close'].shift(1).rolling(window=50).mean()

    # RSI (14)
    price_change = df_stock['Close'].diff()
    rise = price_change.clip(lower=0)
    drop = -price_change.clip(upper=0)
    mean_rise = rise.rolling(window=14).mean()
    mean_drop = drop.rolling(window=14).mean()
    rs_value = mean_rise / mean_drop
    df_stock['RSI'] = 100 - (100 / (1 + rs_value))

    # MACD
    ema_short = df_stock['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df_stock['Close'].ewm(span=26, adjust=False).mean()
    df_stock['MACD'] = ema_short - ema_long

    # VWAP
    df_stock['VWAP'] = (df_stock['Close'] * df_stock['Volume']).cumsum() / df_stock['Volume'].cumsum()

    # 布林带
    window_size = 20
    df_stock['SMA'] = df_stock['Close'].rolling(window=window_size).mean()
    df_stock['Std_dev'] = df_stock['Close'].rolling(window=window_size).std()
    df_stock['Upper_band'] = df_stock['SMA'] + 2 * df_stock['Std_dev']
    df_stock['Lower_band'] = df_stock['SMA'] - 2 * df_stock['Std_dev']

    # OBV
    obv = [0]
    for i in range(1, len(df_stock)):
        if df_stock['Close'].iloc[i] > df_stock['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df_stock['Volume'].iloc[i])
        elif df_stock['Close'].iloc[i] < df_stock['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df_stock['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df_stock['OBV'] = obv

    # ADX
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

    df_stock['+DI'] = 100 * (plus_dm.rolling(window=14).mean() / atr)
    df_stock['-DI'] = 100 * (minus_dm.rolling(window=14).mean() / atr)
    df_stock['ADX'] = (abs(df_stock['+DI'] - df_stock['-DI']) / (df_stock['+DI'] + df_stock['-DI'])).rolling(window=14).mean() * 100

    # ATR
    df_stock['ATR'] = atr

    # 前一天收盘价
    df_stock['Prev_Close'] = df_stock['Close'].shift(1)

    # 删除缺失值
    df_stock = df_stock.dropna()

    return df_stock



def fetch_stock_history(stock_symbol, date_from, date_to):
    """
    获取并处理单个股票的数据

    参数:
        stock_symbol: 股票代码
        date_from: 起始日期
        date_to: 结束日期
    返回:
        处理后的股票数据DataFrame
    """
    # 下载股票数据
    df_raw = yf.download(stock_symbol, start=date_from, end=date_to)  # 无代理
    # df_raw = yf.download(stock_symbol, start=date_from, end=date_to, proxy="http://127.0.0.1:7890")  # 有代理

    # 计算技术指标
    df_final = compute_indicators(df_raw, date_from, date_to)

    return df_final


def clean_up_csv(csv_path):
    df_temp = pd.read_csv(csv_path)

    # 删除第二行和第三行
    df_temp = df_temp.drop([0, 1]).reset_index(drop=True)

    # 重命名列
    df_temp = df_temp.rename(columns={'Price': 'Date'})

    # 保存修改后的文件
    df_temp.to_csv(csv_path, index=False)
    print("所有文件处理完成！")


def process_all():
    """主函数：执行数据收集和处理流程"""
    # 股票分类列表
    symbols = [
        'AAPL', 'MSFT', 'UBER', 'IBM', 'NVDA',  # 科技
        'JPM', 'BAC', 'V', 'MS', 'MA',  # 金融
        'AMZN', 'MCD', 'NIKE', 'TSLA', 'SBUX',  # 消费
        'META', 'NFLX', 'TMUS', 'DIS' 'T'  # 通信服务
        'LLY', 'TMO', 'MRK', 'ABBV', 'GILD', # 医药
        'WM', 'DE', 'BA', 'GE', 'HON',  # 工业
    ]

    # 设置参数
    from_date = '2020-01-01'
    to_date = '2024-01-01'
    keep_features = 9

    # 创建数据文件夹
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)

    # 获取并保存所有股票数据
    print("开始下载和处理股票数据...")
    for code in symbols:
        try:
            print(f"正在处理：{code}")
            df_result = fetch_stock_history(code, from_date, to_date)
            file_full_path = f'{folder_path}/{code}.csv'
            df_result.to_csv(file_full_path)
            clean_up_csv(file_full_path)
            print(f"{code} 处理完成")
        except Exception as err:
            print(f"处理 {code} 时出错: {str(err)}")


if __name__ == "__main__":
    process_all()
