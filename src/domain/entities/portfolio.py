from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class Portfolio:
    """Represents a portfolio of securities with selection and allocation methods."""
    
    def __init__(self, factors: pd.DataFrame, select_by: str = "industry", select_metric: str = "momentum"):
        """
        Initialize a portfolio with selection factors and selection method.
        
        Args:
            factors: DataFrame with securities and their factors (industry, size, etc.)
            select_by: Method to select securities ("industry", "size", "optimize", etc.)
            select_metric: Metric to optimize when using "optimize" selection ("sharpe", "return", "volatility", "drawdown")
        """
        self.factors = factors
        self.select_by = select_by
        self.select_metric = select_metric
        self.weights: Dict[str, float] = {}
    
    def select(self, prices: pd.DataFrame) -> List[str]:
        """
        Select assets from the universe based on the selection method.
        
        Args:
            prices: DataFrame with price history for all securities
            
        Returns:
            List of selected ticker symbols
        """
        # Get the latest available prices
        latest_prices = prices.ffill().iloc[-1]
        
        # Filter assets with valid prices
        valid_assets = latest_prices.dropna().index.tolist()
        
        # Merge with factors
        merged = pd.DataFrame({'price': latest_prices}).join(self.factors)
        merged = merged.dropna(subset=['price'])
        
        # If using optimized selection method
        if self.select_by == "optimize":
            return self._select_optimized(prices, valid_assets)
        
        # Filter by the selection criterion (industry, sector, etc.)
        if self.select_by in merged.columns:
            # Group by the selection criterion
            groups = merged.groupby(self.select_by)
            
            # Select top performing asset from each group
            selected = []
            for name, group in groups:
                if not group.empty:
                    # Calculate simple momentum (end price / start price)
                    if len(prices) > 20:  # Use 20 days of history if available
                        momentum = prices[group.index].iloc[-1] / prices[group.index].iloc[-20]
                        momentum = momentum.dropna()
                        if not momentum.empty:
                            selected.append(momentum.idxmax())
                        else:
                            # Fallback to latest price if momentum can't be calculated
                            group_sorted = group.sort_values('price', ascending=False)
                            selected.append(group_sorted.index[0])
                    else:
                        # Fallback to latest price if not enough history
                        group_sorted = group.sort_values('price', ascending=False)
                        selected.append(group_sorted.index[0])
            
            return [asset for asset in selected if asset in valid_assets]
        
        # If selection criterion not found or invalid, return all valid assets
        return valid_assets
    
    def _select_optimized(self, prices: pd.DataFrame, valid_assets: List[str]) -> List[str]:
        """
        Select assets using optimization on a specific metric.
        
        Args:
            prices: DataFrame with price history
            valid_assets: List of valid assets to consider
            
        Returns:
            List of selected assets based on the optimization metric
        """
        # Filter prices to only include valid assets
        prices = prices[valid_assets].ffill().dropna(axis=1)
        
        # Calculate returns for all assets
        returns = prices.pct_change().dropna()
        
        # Calculate metrics for each asset
        metrics = pd.DataFrame(index=returns.columns)
        
        # Total return
        metrics['total_return'] = (1 + returns).prod() - 1
        
        # Annualized return (assuming daily data)
        days = len(returns)
        years = days / 252
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
        
        # Volatility
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (use 0% risk-free rate for simplicity)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Max drawdown
        for asset in returns.columns:
            asset_returns = returns[asset]
            cumulative = (1 + asset_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            metrics.loc[asset, 'max_drawdown'] = drawdown.min()
        
        # Calculate additional metrics
        
        # Sortino ratio (using downside deviation)
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        downside_deviation = negative_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_deviation
        
        # Calmar ratio (return / abs(max drawdown))
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        
        # Momentum (defined as 3-month return)
        if days >= 63:  # Approximately 3 months of trading days
            momentum_period = min(63, days // 2)
            start_prices = prices.iloc[-momentum_period]
            end_prices = prices.iloc[-1]
            metrics['momentum'] = (end_prices / start_prices) - 1
        else:
            metrics['momentum'] = metrics['total_return']  # Use total return if not enough history
            
        # Group by factors if available
        if self.factors is not None:
            # Select top performing assets based on the metric
            metric_map = {
                'sharpe': 'sharpe_ratio', 
                'return': 'annualized_return',
                'volatility': 'volatility',  # Note: for volatility, lower is better
                'drawdown': 'max_drawdown',  # Note: for drawdown, higher (less negative) is better
                'momentum': 'momentum',
                'sortino': 'sortino_ratio',
                'calmar': 'calmar_ratio'
            }
            
            # Get the column to sort by
            sort_by = metric_map.get(self.select_metric, 'sharpe_ratio')
            
            # Handle metrics where lower is better
            ascending = sort_by in ['volatility', 'max_drawdown']
            
            # Merge metrics with factors
            merged_metrics = metrics.join(self.factors)
            
            # If no factors to group by, select top N assets
            if self.factors.empty or all(col not in merged_metrics.columns for col in self.factors.columns):
                top_n = min(10, len(merged_metrics))  # Select top 10 or fewer if less available
                return merged_metrics.sort_values(by=sort_by, ascending=ascending).head(top_n).index.tolist()
            
            # Group by each factor column
            selected_assets = []
            for col in self.factors.columns:
                if col in merged_metrics.columns:
                    # Group by the factor
                    groups = merged_metrics.groupby(col)
                    
                    # Select top asset from each group
                    for name, group in groups:
                        if not group.empty:
                            # Sort by the metric
                            group_sorted = group.sort_values(by=sort_by, ascending=ascending)
                            if len(group_sorted) > 0:
                                selected_assets.append(group_sorted.index[0])
            
            return list(set(selected_assets))  # Remove duplicates
            
        # If no factors available, select top performing assets by the metric
        sorted_metrics = metrics.sort_values(
            by=metric_map.get(self.select_metric, 'sharpe_ratio'), 
            ascending=sort_by in ['volatility', 'max_drawdown']
        )
        
        # Select top N assets (up to 10)
        top_n = min(10, len(sorted_metrics))
        return sorted_metrics.head(top_n).index.tolist()
    
    def allocate(
        self,
        prices: pd.DataFrame,
        assets: List[str],
        method: str = "optimize",
        target_metric: str = "variance"
    ) -> pd.Series:
        """
        Allocate weights to the selected assets using portfolio optimization.
        
        Args:
            prices: DataFrame with price history
            assets: List of assets to allocate weights to
            method: Allocation method ("optimize" for minimum variance, "equal" for equal weight)
            target_metric: Metric to optimize ("variance", "sharpe", "return")
            
        Returns:
            Series with asset weights
        """
        if not assets:
            return pd.Series()
        
        # Equal weight allocation
        if method == "equal":
            equal_weights = np.array([1.0 / len(assets)] * len(assets))
            weights = pd.Series(equal_weights, index=assets)
            self.weights = weights.to_dict()
            return weights
        
        # Filter and calculate returns
        prices = prices[assets].ffill().dropna(axis=1, how='all')
        assets = list(prices.columns)  # Update assets list after filtering
        
        if not assets:
            return pd.Series()
        
        returns = prices.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Initial weights (equal weight)
        initial_weights = np.array([1.0 / len(assets)] * len(assets))
        
        # Define the objective function based on target metric
        if target_metric == "variance":
            # Minimize portfolio variance
            def _target(x: np.ndarray) -> float:
                portfolio_variance = np.dot(x.T, np.dot(cov_matrix, x))
                return portfolio_variance
        elif target_metric == "sharpe":
            # Maximize Sharpe ratio (negative because we're minimizing)
            def _target(x: np.ndarray) -> float:
                portfolio_return = np.sum(returns.mean() * x) * 252
                portfolio_variance = np.dot(x.T, np.dot(cov_matrix, x)) * 252
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # Negative because we're minimizing
        elif target_metric == "return":
            # Maximize return (negative because we're minimizing)
            def _target(x: np.ndarray) -> float:
                portfolio_return = np.sum(returns.mean() * x) * 252
                return -portfolio_return  # Negative because we're minimizing
        else:
            # Default to minimum variance
            def _target(x: np.ndarray) -> float:
                portfolio_variance = np.dot(x.T, np.dot(cov_matrix, x))
                return portfolio_variance
        
        # Constraint: sum of weights = 1
        def _one_constraint(x: np.ndarray) -> float:
            return np.sum(x) - 1.0
        
        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(len(assets)))
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': _one_constraint},
        )
        
        try:
            # Perform optimization
            result = minimize(
                _target,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            # Check if optimization was successful
            if not result.success:
                # Fallback to equal weights if optimization fails
                weights = pd.Series(initial_weights, index=assets)
            else:
                weights = pd.Series(result.x, index=assets)
                
                # Ensure no extremely small weights (below 0.01%)
                weights[weights < 0.0001] = 0.0
                
                # Renormalize weights if necessary
                if weights.sum() > 0:
                    weights = weights / weights.sum()
        except Exception as e:
            # Fallback to equal weights if optimization fails
            weights = pd.Series(initial_weights, index=assets)
        
        # Store the weights
        self.weights = weights.to_dict()
        
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get the current portfolio weights."""
        return self.weights
    
    def rebalance(self, prices: pd.DataFrame, method: str = "optimize") -> pd.Series:
        """
        Rebalance the portfolio based on current prices.
        
        Args:
            prices: DataFrame with current price data
            method: Allocation method
            
        Returns:
            Series with updated asset weights
        """
        # Select assets
        selected_assets = self.select(prices)
        
        # Allocate weights
        return self.allocate(prices, selected_assets, method=method) 