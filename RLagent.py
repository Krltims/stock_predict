import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from visualization import plot_trading_result

sns.set_style('whitegrid')


class EvolutionaryOptimizer:
    """
    Evolutionary Optimization Strategy

    Args:
        model_params: Parameters of the model
        fitness_func: Fitness evaluation function
        pop_count: Number of individuals in population
        mutation_rate: Standard deviation for mutations
        adapt_rate: Adaptation rate for updates
    """

    def __init__(self, model_params, fitness_func, pop_count, mutation_rate, adapt_rate):
        self.model_params = model_params
        self.fitness_func = fitness_func
        self.pop_count = pop_count
        self.mutation_rate = mutation_rate
        self.adapt_rate = adapt_rate

    def _generate_perturbed_params(self, base_params, perturbations):
        param_variants = []
        for idx, param in enumerate(perturbations):
            variation = self.mutation_rate * param
            param_variants.append(base_params[idx] + variation)
        return param_variants

    def get_current_params(self):
        return self.model_params

    def optimize(self, generations=100, report_interval=1):
        start_time = time.time()
        for gen in range(generations):
            population = []
            fitness_scores = np.zeros(self.pop_count)

            # Create population
            for _ in range(self.pop_count):
                individual = []
                for param in self.model_params:
                    individual.append(np.random.randn(*param.shape))
                population.append(individual)

            # Evaluate fitness
            for k in range(self.pop_count):
                candidate_params = self._generate_perturbed_params(self.model_params, population[k])
                fitness_scores[k] = self.fitness_func(candidate_params)

            # Normalize scores
            fitness_scores = (fitness_scores - np.mean(fitness_scores)) / (np.std(fitness_scores) + 1e-7)

            # Update parameters
            for idx, param in enumerate(self.model_params):
                perturbations_matrix = np.array([p[idx] for p in population])
                self.model_params[idx] = (
                        param + self.adapt_rate / (self.pop_count * self.mutation_rate)
                        * np.dot(perturbations_matrix.T, fitness_scores).T
                )

            if (gen + 1) % report_interval == 0:
                print(f'Generation {gen + 1}. Fitness: {self.fitness_func(self.model_params):.3f}')
        print(f'Optimization completed in {time.time() - start_time:.2f} seconds')


class NeuralNetwork:
    """
    Neural Network Model

    Args:
        input_dim: Input dimension size
        hidden_dim: Hidden layer size
        output_dim: Output dimension size
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.parameters = [
            np.random.randn(input_dim, hidden_dim),
            np.random.randn(hidden_dim, output_dim),
            np.random.randn(1, hidden_dim),
        ]

    def forward(self, input_data):
        hidden_layer = np.dot(input_data, self.parameters[0]) + self.parameters[-1]
        activated_hidden_layer = np.tanh(hidden_layer) 
        output = np.dot(activated_hidden_layer, self.parameters[1])
        return output

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, new_params):
        self.parameters = new_params


class TradingStrategy:
    """
    Stock Trading Strategy Implementation

    Args:
        predictor: Prediction model
        lookback_period: Historical window size
        price_data: Price sequence
        step_size: Step interval
        init_capital: Starting capital
        stock_symbol: Ticker symbol
        output_path: Directory for saving results
    """
    POPULATION_COUNT = 15
    MUTATION_FACTOR = 0.1
    ADAPTATION_RATE = 0.03

    def __init__(self, predictor, lookback_period, price_data, step_size, init_capital, stock_symbol, output_path):
        self.predictor = predictor
        self.lookback_period = lookback_period
        self.half_window = lookback_period // 2
        self.price_data = price_data
        self.step_size = step_size
        self.init_capital = init_capital
        self.stock_symbol = stock_symbol
        self.output_path = output_path
        self.optimizer = EvolutionaryOptimizer(
            self.predictor.get_parameters(),
            self.evaluate_performance,
            self.POPULATION_COUNT,
            self.MUTATION_FACTOR,
            self.ADAPTATION_RATE,
        )

    def decide_action(self, state_sequence):
        decision = self.predictor.forward(np.array(state_sequence))
        return np.argmax(decision[0])

    def get_current_state(self, time_idx):
        window = self.lookback_period + 1
        start_idx = time_idx - window + 1
        data_block = self.price_data[start_idx: time_idx + 1] if start_idx >= 0 else \
            -start_idx * [self.price_data[0]] + self.price_data[0: time_idx + 1]
        state = []
        for i in range(window - 1):
            state.append(data_block[i + 1] - data_block[i])
        return np.array([state])

    def evaluate_performance(self, model_parameters):
        capital = self.init_capital
        starting_capital = capital
        self.predictor.set_parameters(model_parameters)
        current_state = self.get_current_state(0)
        holdings = []

        for t in range(0, len(self.price_data) - 1, self.step_size):
            action = self.decide_action(current_state)
            next_state = self.get_current_state(t + 1)

            if action == 1 and starting_capital >= self.price_data[t]:
                holdings.append(self.price_data[t])
                starting_capital -= self.price_data[t]

            elif action == 2 and len(holdings):
                purchase_price = holdings.pop(0)
                starting_capital += self.price_data[t]

            current_state = next_state
        return ((starting_capital - capital) / capital) * 100

    def train_strategy(self, training_rounds, checkpoint_interval):
        self.optimizer.optimize(training_rounds, checkpoint_interval)

    def execute_trades(self, save_directory):
        capital = self.init_capital
        current_state = self.get_current_state(0)
        starting_capital = capital
        buy_points = []
        sell_points = []
        holdings = []
        trade_history = []

        for t in range(0, len(self.price_data) - 1, self.step_size):
            action = self.decide_action(current_state)
            next_state = self.get_current_state(t + 1)

            if action == 1 and capital >= self.price_data[t]:
                holdings.append(self.price_data[t])
                capital -= self.price_data[t]
                buy_points.append(t)
                trade_history.append({
                    'day': t,
                    'operate': 'buy',
                    'price': self.price_data[t],
                    'investment': 0,
                    'total_balance': capital
                })

            elif action == 2 and len(holdings):
                purchase_price = holdings.pop(0)
                capital += self.price_data[t]
                sell_points.append(t)
                try:
                    roi = ((self.price_data[t] - purchase_price) / purchase_price) * 100
                except:
                    roi = 0
                trade_history.append({
                    'day': t,
                    'operate': 'sell',
                    'price': self.price_data[t],
                    'investment': roi,
                    'total_balance': capital
                })

            current_state = next_state

        # Save transaction records
        transaction_df = pd.DataFrame(trade_history)
        os.makedirs(f'{save_directory}/transactions', exist_ok=True)
        transaction_df.to_csv(f'{save_directory}/transactions/{self.stock_symbol}_transactions.csv', index=False)

        roi_percentage = ((capital - starting_capital) / starting_capital) * 100
        net_profit = capital - starting_capital
        return buy_points, sell_points, net_profit, roi_percentage


def process_stock(ticker, save_dir, model_type, window_size=30, initial_money=10000, iterations=500):
    try:
        # Load prediction data based on model_type
        prediction_file = f'{save_dir}/predictions/{ticker}_{model_type}_predictions.pkl'
        print(f"\nProcessing {ticker} with agent based on {model_type} predictions")
        df = pd.read_pickle(prediction_file)
        price_sequence = df.Prediction.values.tolist()

        # Configure parameters
        window_size = window_size
        step_size = 1
        initial_money = initial_money

        # Create a unique symbol for this run to avoid overwriting outputs
        stock_symbol_with_model = f"{ticker}_{model_type}"

        # Initialize components
        prediction_model = NeuralNetwork(input_dim=window_size, hidden_dim=128, output_dim=3)
        trading_agent = TradingStrategy(
            predictor=prediction_model,
            lookback_period=window_size,
            price_data=price_sequence,
            step_size=step_size,
            init_capital=initial_money,
            stock_symbol=stock_symbol_with_model,  # Use unique symbol
            output_path=save_dir
        )

        # Train the strategy
        trading_agent.train_strategy(training_rounds=iterations, checkpoint_interval=10)

        # Execute trading and collect results
        buy_signals, sell_signals, profit, roi = trading_agent.execute_trades(save_dir)

        # Generate visualization
        plot_trading_result(stock_symbol_with_model, price_sequence, buy_signals, sell_signals, profit, roi, save_dir)

        return {
            'total_gains': profit,
            'investment_return': roi,
            'trades_buy': len(buy_signals),
            'trades_sell': len(sell_signals)
        }

    except Exception as e:
        print(f"Error processing {ticker} with {model_type} predictions: {e}")
        return None


def main():
    """Main execution function"""
    stock_list = [
        'MSFT', 'AAPL', 'UBER', 'IBM', 'NVDA',
        'JPM', 'BAC', 'V', 'MS', 'MA',
        'AMZN', 'MCD', 'NKE', 'TSLA', 'SBUX',
        'META', 'NFLX', 'TMUS', 'DIS', 'T',
        'LLY', 'TMO', 'MRK', 'ABBV', 'GILD',
        'WM', 'DE', 'BA', 'GE', 'HON',
    ]
    output_directory = 'results'
    model_types = ['LSTM', 'GRU']

    for symbol in stock_list:
        for model_type in model_types:
            process_stock(symbol, output_directory, model_type)


if __name__ == "__main__":
    main()
