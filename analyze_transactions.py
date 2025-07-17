import os
import pandas as pd

def analyze_transactions(folder_path='results/transactions'):
    """分析交易记录，区分LSTM和GRU模型，计算关键指标"""
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # 提取股票代码和模型类型
            parts = filename.replace('_transactions.csv', '').rsplit('_', 1)
            if len(parts) != 2:
                continue  # 跳过格式不符的文件
            ticker, model_type = parts

            df = pd.read_csv(os.path.join(folder_path, filename))

            # 初始资金
            initial_money = 10000
            final_balance = df['total_balance'].iloc[-1]
            total_gains = final_balance - initial_money
            returns = (total_gains / initial_money) * 100

            buy_count = len(df[df['operate'] == 'buy'])
            sell_count = len(df[df['operate'] == 'sell'])
            total_trades = buy_count + sell_count

            profitable_trades = len(df[df['investment'] > 0])
            win_rate = (profitable_trades / sell_count * 100) if sell_count > 0 else 0

            balance_series = df['total_balance']
            rolling_max = balance_series.expanding().max()
            drawdown = (rolling_max - balance_series) / rolling_max * 100
            max_drawdown = drawdown.max()

            results.append({
                'Stock': ticker,
                'Model': model_type.upper(),
                'Total Gains ($)': round(total_gains, 2),
                'Returns (%)': round(returns, 2),
                'Total Trades': total_trades,
                'Win Rate (%)': round(win_rate, 2),
                'Max Drawdown (%)': round(max_drawdown, 2)
            })

    results_df = pd.DataFrame(results)
    os.makedirs('results/output', exist_ok=True)
    results_df.to_csv('results/output/prediction_metrics.csv', index=False)

    print("\nTrading Performance Summary:")
    print("=" * 50)
    print(results_df.groupby('Model').describe())

    print("\nDetailed Results by Stock and Model:")
    print("=" * 50)
    print(results_df)

    return results_df

if __name__ == "__main__":
    results = analyze_transactions()