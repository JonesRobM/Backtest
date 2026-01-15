import pandas as pd

class Portfolio:
    """
    Represents a portfolio with a target allocation and a rebalancing strategy.
    """
    def __init__(self, target_weights, rebalance_frequency='monthly'):
        """
        Initializes the Portfolio.

        Args:
            target_weights (dict): Tickers and their target weights (e.g., {'AAPL': 0.6, 'GOOG': 0.4}).
                                   Weights should be floats that sum to 1.0.
            rebalance_frequency (str): How often to rebalance. 
                                       Options: 'monthly', 'quarterly', 'annually', or None for buy-and-hold.
        """
        if not isinstance(target_weights, dict) or not all(isinstance(k, str) for k in target_weights.keys()):
            raise TypeError("target_weights must be a dictionary of string tickers.")
        
        if round(sum(target_weights.values()), 5) != 1.0:
            raise ValueError(f"Target weights must sum to 1.0. Current sum: {sum(target_weights.values())}")

        self.target_weights = target_weights
        self.rebalance_frequency = rebalance_frequency
        self._last_rebalance_date = None  # Internal state to track rebalancing dates

    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """
        Determines if the portfolio should be rebalanced based on the current date
        and the chosen rebalancing frequency.

        Args:
            current_date (pd.Timestamp): The current date in the backtest simulation.

        Returns:
            bool: True if a rebalance is needed, False otherwise.
        """
        # No rebalancing for "buy and hold" strategy
        if self.rebalance_frequency is None:
            return False

        # If it's the first time we're checking, we don't rebalance yet.
        # The initial buy on day 1 acts as the first "balancing".
        if self._last_rebalance_date is None:
            self._last_rebalance_date = current_date
            return False

        # Check against the specified frequency
        if self.rebalance_frequency == 'monthly':
            # Rebalance if the month is different from the last rebalance month
            if current_date.month != self._last_rebalance_date.month:
                self._last_rebalance_date = current_date
                return True
        elif self.rebalance_frequency == 'quarterly':
            # Rebalance if the quarter is different
            if current_date.quarter != self._last_rebalance_date.quarter:
                self._last_rebalance_date = current_date
                return True
        elif self.rebalance_frequency == 'annually':
            # Rebalance if the year is different
            if current_date.year != self._last_rebalance_date.year:
                self._last_rebalance_date = current_date
                return True
        
        return False