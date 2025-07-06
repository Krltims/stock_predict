import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from visualization import plot_trading_result
sns.set()

class Deep_Evolution_Strategy:
    """
    Deep Evolution Strategy Class
    
    Parameters:
        weights: Model weights
        reward_function: Reward function
        population_size: Population size
        sigma: Perturbation standard deviation
        learning_rate: Learning rate
    """
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        """Generate perturbed weights"""
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        """Get current weights"""
        return self.weights

    def train(self, epoch=100, print_every=1):
        """
        Train the model
        
        Parameters:
            epoch: Number of training epochs
            print_every: Print frequency
        """
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            # Generate population
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            # Calculate reward for each individual
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(weights_population)
            # Normalize rewards
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            # Update weights
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print('iter %d. reward: %f' % (i + 1, self.reward_function(self.weights)))
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    """
    Neural Network Model Class
    
    Parameters:
        input_size: Input dimension
        layer_size: Hidden layer size
        output_size: Output dimension
    """
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        """Prediction function"""
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        """Get model weights"""
        return self.weights

    def set_weights(self, weights):
        """Set model weights"""
        self.weights = weights


class Agent:
    """
    Trading Agent Class
    
    Parameters:
        model: Prediction model
        window_size: Time window size
        trend: Price sequence
        skip: Skip steps
        initial_money: Initial capital
        ticker: Stock symbol
    """
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, window_size, trend, skip, initial_money, ticker, save_dir):
        self.model = model
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.ticker = ticker
        self.save_dir = save_dir
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        """Choose action based on current state"""
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def get_state(self, t):
        """Get current state"""
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def get_reward(self, weights):
        """Calculate reward value"""
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        state = self.get_state(0)
        inventory = []
        
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]

            state = next_state
        return ((starting_money - initial_money) / initial_money) * 100

    def fit(self, iterations, checkpoint):
        """Train the agent"""
        self.es.train(iterations, print_every=checkpoint)

    def buy(self, save_dir):
        """Execute trading strategy"""
        initial_money = self.initial_money
        state = self.get_state(0)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        transaction_history = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                transaction_history.append({
                    'day': t,
                    'operate': 'buy',
                    'price': self.trend[t],
                    'investment': 0,
                    'total_balance': initial_money
                })

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                transaction_history.append({
                    'day': t,
                    'operate': 'sell',
                    'price': self.trend[t],
                    'investment': invest,
                    'total_balance': initial_money
                })

            state = next_state

        # Save transaction history
        df_transaction = pd.DataFrame(transaction_history)
        os.makedirs(f'{save_dir}/transactions', exist_ok=True)
        df_transaction.to_csv(f'{save_dir}/transactions/{self.ticker}_transactions.csv', index=False)

        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest


def process_stock(ticker, save_dir, window_size = 30, initial_money = 10000, iterations=500):
    try:
        # Read prediction data
        df = pd.read_pickle(f'{save_dir}/predictions/{ticker}_predictions.pkl')
        print(f"\nProcessing {ticker}")
        close = df.Prediction.values.tolist()

        # Set parameters
        window_size = window_size
        skip = 1
        initial_money = initial_money

        # Create model and agent
        model = Model(input_size=window_size, layer_size=500, output_size=3)
        agent = Agent(model=model, window_size=window_size, trend=close, 
                     skip=skip, initial_money=initial_money, ticker=ticker, save_dir=save_dir)
        
        # Train agent
        agent.fit(iterations=iterations, checkpoint=10)

        # Execute trading and get results
        states_buy, states_sell, total_gains, invest = agent.buy(save_dir)

        # Use visualization tool to draw trading chart
        plot_trading_result(ticker, close, states_buy, states_sell, total_gains, invest, save_dir)
        
        return {
            'total_gains': total_gains,
            'investment_return': invest,
            'trades_buy': len(states_buy),
            'trades_sell': len(states_sell)
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def main():
    """Main function: Execute all stock trading strategy"""
    # Stock list
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # Technology
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # Finance
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # Pharmaceutical
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # Energy
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # Consumer
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # Industrial
    ]
    save_dir = 'results'
    # Process each stock
    for ticker in tickers:
        process_stock(ticker, save_dir)


if __name__ == "__main__":
    main()