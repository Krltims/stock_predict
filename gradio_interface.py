import gradio as gr
import pandas as pd
import torch
import os
from PIL import Image
import warnings
import yfinance as yf
from model import predict, format_feature
from RLagent import process_stock
from datetime import datetime
from process_stock_data import get_stock_data, clean_csv_files

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用相对路径并确保Windows兼容性
SAVE_DIR = os.path.join(os.getcwd(), 'tmp', 'gradio')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'pic'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'ticker'), exist_ok=True)

# 自定义CSS样式 - 清新蓝白配色
custom_css = """
.gradio-container {
    background-color: #f0f7ff;
}

.gr-button {
    background-color: #4a90e2;
    color: white;
    border: none;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.gr-button:hover {
    background-color: #357abd;
    transform: translateY(-2px);
}

.gr-button:active {
    transform: translateY(0);
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}

.gr-tab-item {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.gr-slider input[type=range]::-webkit-slider-thumb {
    background: #4a90e2;
}

.gr-textbox, .gr-number, .gr-dropdown {
    border: 1px solid #bdc3c7;
    border-radius: 6px;
    transition: border 0.3s ease;
}

.gr-textbox:focus, .gr-number:focus, .gr-dropdown:focus {
    border-color: #4a90e2;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}

.status-success {
    color: #27ae60;
}

.status-error {
    color: #e74c3c;
}

.gr-gallery {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.gr-dataframe {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
"""

def get_data(ticker, start_date, end_date, progress=gr.Progress()):
    data_folder = os.path.join(SAVE_DIR, 'ticker')
    temp_path = f'{data_folder}/{ticker}.csv'

    try:
        # 获取并保存所有股票数据
        progress(0, desc="Start obtaining stock data...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        progress(0.4, desc="Calculate technical indicators...")
        stock_data.to_csv(temp_path)
        progress(0.7, desc="Processing data format...")
        clean_csv_files(temp_path)
        progress(1.0, desc="Data acquisition completed")
        return temp_path, f'<span class="status-success">Data acquisition successful</span>'
    except Exception as e:
        return None, f'<span class="status-error">Error in obtaining data: {str(e)}</span>'

def process_and_predict(temp_csv_path, model_type,
                       # LSTM参数
                       lstm_epochs, lstm_batch, lstm_learning_rate,
                       # GRU参数
                       gru_epochs, gru_batch, gru_learning_rate,
                       # 通用参数
                       window_size, initial_money, agent_iterations, save_dir):
    if not temp_csv_path:
        return [None] * 9

    try:
        ticker = os.path.basename(temp_csv_path).split('_')[0]
        stock_data = pd.read_csv(temp_csv_path)
        stock_features = format_feature(stock_data)

        # 根据选择的模型调用不同的预测函数
        if model_type == "LSTM":
            metrics = predict(
                save_dir=save_dir,
                ticker_name=ticker,
                stock_data=stock_data,
                stock_features=stock_features,
                model_type=model_type,  # 传递模型类型参数
                epochs=lstm_epochs,
                batch_size=lstm_batch,
                learning_rate=lstm_learning_rate
            )
        else:  # GRU模型
            metrics = predict(
                save_dir=save_dir,
                ticker_name=ticker,
                stock_data=stock_data,
                stock_features=stock_features,
                model_type=model_type,  # 传递模型类型参数
                epochs=gru_epochs,
                batch_size=gru_batch,
                learning_rate=gru_learning_rate
            )

        trading_results = process_stock(
            ticker,
            save_dir,
            window_size=window_size,
            initial_money=initial_money,
            iterations=agent_iterations
        )

        prediction_plot = Image.open(f"{save_dir}/pic/predictions/{ticker}_prediction.png")
        loss_plot = Image.open(f"{save_dir}/pic/loss/{ticker}_loss.png")
        earnings_plot = Image.open(f"{save_dir}/pic/earnings/{ticker}_cumulative.png")
        trades_plot = Image.open(f"{save_dir}/pic/trades/{ticker}_trades.png")
        transactions_df = pd.read_csv(f"{save_dir}/transactions/{ticker}_transactions.csv")

        return [
            [prediction_plot, loss_plot, earnings_plot, trades_plot],
            metrics['accuracy'] * 100,
            metrics['rmse'],
            metrics['mae'],
            trading_results['total_gains'],
            trading_results['investment_return'],
            trading_results['trades_buy'],
            trading_results['trades_sell'],
            transactions_df
        ]
    except Exception as e:
        print(f"process error: {str(e)}")
        return [None] * 9

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Intelligent Stock Prediction and Trading Agent")

    save_dir_state = gr.State(value=SAVE_DIR)
    temp_csv_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(label="Stock code", placeholder="Enter stock symbol (e.g., AAPL)")
        with gr.Column(scale=2):
            start_date = gr.Textbox(
                label="Start date (YYYY-MM-DD)",
                value=(datetime.now().replace(year=datetime.now().year-4).strftime('%Y-%m-%d'))
            )
        with gr.Column(scale=2):
            end_date = gr.Textbox(
                label="End date (YYYY-MM-DD)",
                value=datetime.now().strftime('%Y-%m-%d')
            )
        with gr.Column(scale=1):
            fetch_button = gr.Button("Get data", variant="primary")

    with gr.Row():
        status_output = gr.HTML(label="Status information", interactive=False)

    with gr.Row():
        data_file = gr.File(label="Download stock data", visible=True, interactive=False)

    # 修改为单选下拉框
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["LSTM", "GRU"],
            label="Prediction Model",
            value="LSTM",  # 默认选择LSTM
            info="Select the model to use for stock price prediction",
            multiselect=False  # 关闭多选
        )

    with gr.Tabs():
        with gr.TabItem("LSTM Prediction parameters") as lstm_tab:
            with gr.Column():
                lstm_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10,
                                      label="LSTM Training rounds")
                lstm_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16,
                                     label="LSTM Batch size")
                lstm_learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001,
                                        step=0.0001, label="LSTM Learning rate")

        with gr.TabItem("GRU Prediction parameters") as gru_tab:
            with gr.Column():
                gru_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10,
                                      label="GRU Training rounds")
                gru_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16,
                                     label="GRU Batch size")
                gru_learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001,
                                        step=0.0001, label="GRU Learning rate")


        with gr.TabItem("Trading agent parameters"):
            with gr.Column():
                window_size = gr.Slider(minimum=10, maximum=100, value=30, step=5,
                                      label="Time window size")
                initial_money = gr.Number(value=10000, label="Initial investment amount ($)")
                agent_iterations = gr.Slider(minimum=100, maximum=1000, value=500,
                                          step=50, label="Agent training iterations")

    with gr.Row():
        train_button = gr.Button("Start training", variant="primary", interactive=False)

    with gr.Row():
        output_gallery = gr.Gallery(label="Analysis results visualization", show_label=True,
                                  elem_id="gallery", columns=4, rows=1,
                                  height="auto", object_fit="contain")

    # 新增输出指标显示区域
    with gr.Row():
        with gr.Column():
            accuracy = gr.Number(label="Accuracy (%)")
            rmse = gr.Number(label="RMSE")
            mae = gr.Number(label="MAE")
        with gr.Column():
            total_gains = gr.Number(label="Total Gains ($)")
            investment_return = gr.Number(label="Return Rate (%)")
            trades_buy = gr.Number(label="Buy Trades")
            trades_sell = gr.Number(label="Sell Trades")

    with gr.Row():
        transactions_df = gr.Dataframe(label="Transaction History")


    def update_interface(csv_path):
        return (
            csv_path if csv_path else None,
            gr.update(interactive=bool(csv_path))
        )

    # 根据选择的模型显示/隐藏对应的参数标签页
    def update_model_tabs(model_type):
        if model_type == "LSTM":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    # 模型选择变化时更新标签页显示
    model_selector.change(
        fn=update_model_tabs,
        inputs=[model_selector],
        outputs=[lstm_tab, gru_tab]
    )

    fetch_result = fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date, end_date],
        outputs=[temp_csv_state, status_output]
    )

    fetch_result.then(
        update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file, train_button]
    )

    train_button.click(
        fn=process_and_predict,
        inputs=[
            temp_csv_state,
            model_selector,
            # LSTM参数
            lstm_epochs, lstm_batch, lstm_learning_rate,
            # GRU参数
            gru_epochs, gru_batch, gru_learning_rate,
            # 通用参数
            window_size, initial_money, agent_iterations, save_dir_state
        ],
        outputs=[
            output_gallery,
            accuracy,
            rmse,
            mae,
            total_gains,
            investment_return,
            trades_buy,
            trades_sell,
            transactions_df
        ]
    )

demo.launch(server_port=7860, share=True, enable_queue=True)