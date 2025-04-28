from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        
        # Calculate CVaR (Conditional Value at Risk)
        confidence_level = 0.95
        for asset in returns.columns:
            asset_returns = returns[asset]
            sorted_returns = asset_returns.sort_values()
            cutoff_index = int(np.ceil(len(sorted_returns) * (1 - confidence_level)))
            cvar = sorted_returns[:cutoff_index].mean()
            metrics.loc[asset, 'cvar'] = cvar
            
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
                'calmar': 'calmar_ratio',
                'cvar': 'cvar'  # Add CVaR to the map
            }
            
            # Get the column to sort by
            sort_by = metric_map.get(self.select_metric, 'sharpe_ratio')
            
            # Handle metrics where lower is better
            ascending = sort_by in ['volatility', 'max_drawdown', 'cvar']
            
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
            ascending=sort_by in ['volatility', 'max_drawdown', 'cvar']
        )
        
        # Select top N assets (up to 10)
        top_n = min(10, len(sorted_metrics))
        return sorted_metrics.head(top_n).index.tolist()
    
    def _calculate_cvar(self, returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
        """
        Calculate Conditional Value at Risk (CVaR) for each asset.
        
        Args:
            returns: DataFrame with return history
            confidence_level: Confidence level for CVaR calculation (default: 0.95)
            
        Returns:
            Series with CVaR values for each asset
        """
        cvar_values = pd.Series(index=returns.columns)
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) > 0:
                sorted_returns = asset_returns.sort_values()
                cutoff_index = int(np.ceil(len(sorted_returns) * (1 - confidence_level)))
                if cutoff_index > 0:
                    cvar = sorted_returns[:cutoff_index].mean()
                    cvar_values[asset] = cvar
                else:
                    cvar_values[asset] = asset_returns.min()
        
        return cvar_values
    
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
            method: Allocation method ("optimize" for minimum variance, "equal" for equal weight,
                    "inverse_vol" for inverse volatility, "cvar" for inverse CVaR)
            target_metric: Metric to optimize ("variance", "sharpe", "return", "cvar_ratio")
            
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
        
        # Inverse volatility allocation
        if method == "inverse_vol":
            vols = returns.std()
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()
            self.weights = weights.to_dict()
            return weights
        
        # Inverse CVaR allocation
        if method == "cvar":
            cvar_values = self._calculate_cvar(returns)
            # Convert CVaR to positive values for weighting
            abs_cvar = cvar_values.abs()
            inv_cvar = 1.0 / abs_cvar
            weights = inv_cvar / inv_cvar.sum()
            self.weights = weights.to_dict()
            return weights
        
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
        elif target_metric == "cvar_ratio":
            # Maximize return to CVaR ratio (negative because we're minimizing)
            def _target(x: np.ndarray) -> float:
                # Calculate portfolio return
                portfolio_return = np.sum(returns.mean() * x) * 252
                
                # Calculate portfolio CVaR
                portfolio_returns = returns.dot(x)
                sorted_returns = portfolio_returns.sort_values()
                confidence_level = 0.95
                cutoff_index = int(np.ceil(len(sorted_returns) * (1 - confidence_level)))
                if cutoff_index > 0:
                    portfolio_cvar = sorted_returns[:cutoff_index].mean()
                    if portfolio_cvar < 0:  # Ensure CVaR is negative for ratio calculation
                        cvar_ratio = portfolio_return / abs(portfolio_cvar)
                        return -cvar_ratio  # Negative because we're minimizing
                
                # Fallback to sharpe ratio if CVaR calculation fails
                portfolio_variance = np.dot(x.T, np.dot(cov_matrix, x)) * 252
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio
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
    
    def rebalance(self, prices: pd.DataFrame, method: str = "optimize", target_metric: str = "variance") -> pd.Series:
        """
        Rebalance the portfolio based on current prices.
        
        Args:
            prices: DataFrame with current price data
            method: Allocation method
            target_metric: Metric to optimize
            
        Returns:
            Series with updated asset weights
        """
        # Select assets
        selected_assets = self.select(prices)
        
        # Allocate weights
        return self.allocate(prices, selected_assets, method=method, target_metric=target_metric)
    
    def calculate_rebalance_orders(
        self,
        current_positions: pd.DataFrame,
        prices: pd.DataFrame,
        cash: float = 0,
        method: str = "optimize",
        target_metric: str = "variance",
        threshold: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculate rebalance orders to reach target weights.
        
        Args:
            current_positions: DataFrame with current positions (tickers, quantities, values)
            prices: DataFrame with current price data
            cash: Available cash
            method: Weight allocation method
            target_metric: Metric to optimize for weight allocation
            threshold: Minimum deviation threshold to trigger rebalancing (default: 5%)
            
        Returns:
            Tuple of (orders DataFrame, target weights Series)
        """
        logger.info("Calculating rebalance orders")
        
        # Calculate current portfolio value
        if 'market_value' in current_positions.columns:
            portfolio_value = current_positions['market_value'].sum() + cash
        elif 'current_price' in current_positions.columns and 'quantity' in current_positions.columns:
            current_positions['market_value'] = current_positions['quantity'] * current_positions['current_price']
            portfolio_value = current_positions['market_value'].sum() + cash
        else:
            raise ValueError("Current positions must contain market_value or both quantity and current_price")
        
        logger.info(f"Current portfolio value: {portfolio_value:.2f}")
        
        # Calculate current weights
        if portfolio_value > 0:
            current_positions['weight'] = current_positions['market_value'] / portfolio_value
        else:
            current_positions['weight'] = 0
        
        # Get target weights
        selected_assets = self.select(prices)
        target_weights = self.allocate(prices, selected_assets, method=method, target_metric=target_metric)
        
        # Merge current positions with target weights
        all_assets = set(current_positions.index) | set(target_weights.index)
        rebalance_df = pd.DataFrame(index=all_assets)
        
        # Add current positions
        rebalance_df['current_quantity'] = current_positions.get('quantity', pd.Series(0, index=all_assets))
        rebalance_df['current_price'] = current_positions.get('current_price', pd.Series(0, index=all_assets))
        rebalance_df['current_value'] = current_positions.get('market_value', pd.Series(0, index=all_assets))
        rebalance_df['current_weight'] = current_positions.get('weight', pd.Series(0, index=all_assets))
        
        # Add target weights and fill NaN values
        rebalance_df['target_weight'] = target_weights
        rebalance_df = rebalance_df.fillna(0)
        
        # Calculate target values and differences
        rebalance_df['target_value'] = rebalance_df['target_weight'] * portfolio_value
        rebalance_df['value_difference'] = rebalance_df['target_value'] - rebalance_df['current_value']
        rebalance_df['weight_difference'] = rebalance_df['target_weight'] - rebalance_df['current_weight']
        
        # Filter based on threshold
        rebalance_df['needs_rebalance'] = abs(rebalance_df['weight_difference']) > threshold
        
        # Calculate lots to trade if lot size information is available
        if 'lot' in current_positions.columns:
            rebalance_df['lot'] = current_positions.get('lot', pd.Series(1, index=all_assets)).fillna(1)
            rebalance_df['order_quantity'] = (rebalance_df['value_difference'] / rebalance_df['current_price'] / rebalance_df['lot']).round().astype(int) * rebalance_df['lot']
        else:
            # Assume lot size of 1 if not provided
            rebalance_df['order_quantity'] = (rebalance_df['value_difference'] / rebalance_df['current_price']).round().astype(int)
        
        # Create orders DataFrame with necessary information
        orders = rebalance_df[rebalance_df['needs_rebalance']].copy()
        orders['is_buy'] = orders['order_quantity'] > 0
        orders['order_value'] = orders['order_quantity'] * orders['current_price']
        
        # Sort by absolute order value
        orders = orders.sort_values(by='order_value', ascending=False)
        
        logger.info(f"Generated {len(orders)} rebalance orders")
        
        return orders, target_weights
    
    def get_mean_absolute_percentage_error(self, current_weights: pd.Series, target_weights: pd.Series) -> float:
        """
        Calculate Mean Absolute Percentage Error between current and target weights.
        
        Args:
            current_weights: Series with current portfolio weights
            target_weights: Series with target portfolio weights
            
        Returns:
            MAPE value
        """
        # Align the weights to have the same indices
        combined = pd.concat([current_weights, target_weights], axis=1).fillna(0)
        current = combined[0].values
        target = combined[1].values
        
        # Avoid division by zero
        epsilon = np.finfo(np.float64).eps
        
        # Calculate MAPE
        mape = np.abs(target - current) / np.maximum(np.abs(target), epsilon)
        return float(np.mean(mape))
    
    def prepare_rebalance_positions(
        self,
        raw_positions: pd.DataFrame,
        target_weights: pd.Series,
        additional_info: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare positions data for rebalancing by adding target weights and calculating differences.
        
        Args:
            raw_positions: DataFrame with raw position data
            target_weights: Series with target weights
            additional_info: DataFrame with additional information like lot sizes and tickers
            
        Returns:
            DataFrame with prepared positions
        """
        if additional_info is None:
            additional_info = pd.DataFrame()
            
        # Set index to figi or ticker
        if 'figi' in raw_positions.columns:
            positions = raw_positions.set_index('figi')
        elif 'ticker' in raw_positions.columns:
            positions = raw_positions.set_index('ticker')
        else:
            positions = raw_positions.copy()
            
        # Combine indices from positions and target weights
        index = positions.index.union(target_weights.index)
        
        # Ensure necessary columns
        for col in ['lot', 'ticker', 'current_price']:
            if col not in positions.columns:
                positions[col] = np.nan
        
        # Add cash as a special position if needed
        if 'USD000UTSTOM' not in index and 'RUB000UTSTOM' not in index:
            # Add a placeholder for cash
            cash_symbol = 'RUB000UTSTOM'  # or 'USD000UTSTOM' for USD
            if cash_symbol not in positions.index:
                positions.loc[cash_symbol, 'current_price'] = 1.0
                index = index.append(pd.Index([cash_symbol]))
                
        # Fill in data from additional_info and defaults
        positions = positions.reindex(index).fillna(additional_info).fillna(0)
        
        # Calculate market value
        if 'market_value' not in positions.columns:
            positions['market_value'] = positions['quantity'] * positions['current_price']
        
        # Calculate current weights
        total_value = positions['market_value'].sum()
        if total_value > 0:
            positions['weight'] = positions['market_value'] / total_value
        else:
            positions['weight'] = 0
            
        # Add target weights
        positions['target_weight'] = target_weights.reindex(index).fillna(0)
        
        # Calculate differences
        positions['delta_weight'] = positions['target_weight'] - positions['weight']
        positions['target_market_value'] = positions['target_weight'] * total_value
        positions['delta_market_value'] = positions['target_market_value'] - positions['market_value']
        
        # Remove cash from trading calculations
        for cash_sym in ['USD000UTSTOM', 'RUB000UTSTOM']:
            if cash_sym in positions.index:
                positions = positions.drop(index=[cash_sym])
                
        # Calculate order quantities in lots
        positions['delta_quantity_for_order'] = (
            positions['delta_market_value'] / positions['current_price'] / positions['lot']
        ).astype(int)
        
        # Filter out positions that don't need rebalancing
        positions = positions[positions['delta_quantity_for_order'] != 0].sort_values('delta_market_value')
        
        return positions
    
    def execute_rebalance(
        self,
        orders: pd.DataFrame,
        execute_order_function: Callable = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute rebalance orders.
        
        Args:
            orders: DataFrame with orders from calculate_rebalance_orders
            execute_order_function: Function to execute orders (receives symbol, quantity, is_buy)
            dry_run: If True, only simulate execution
            
        Returns:
            Dictionary with execution results
        """
        if execute_order_function is None or dry_run:
            logger.info("Dry run mode - not executing orders")
            return {
                "executed": [],
                "skipped": list(orders.index),
                "total_value": orders['order_value'].sum(),
                "dry_run": True
            }
            
        executed = []
        skipped = []
        total_executed_value = 0
        
        for symbol, order in orders.iterrows():
            try:
                quantity = abs(int(order['order_quantity']))
                is_buy = order['is_buy']
                
                if quantity == 0:
                    continue
                    
                logger.info(f"Executing {'BUY' if is_buy else 'SELL'} order for {symbol}: {quantity} units")
                
                # Execute the order using the provided function
                result = execute_order_function(
                    symbol=symbol,
                    quantity=quantity,
                    is_buy=is_buy
                )
                
                executed.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "is_buy": is_buy,
                    "value": order['order_value'],
                    "result": result
                })
                
                total_executed_value += abs(order['order_value'])
                
            except Exception as e:
                logger.error(f"Error executing order for {symbol}: {e}")
                skipped.append(symbol)
                
        logger.info(f"Executed {len(executed)} orders worth {total_executed_value:.2f}")
        logger.info(f"Skipped {len(skipped)} orders")
        
        return {
            "executed": executed,
            "skipped": skipped,
            "total_value": total_executed_value,
            "dry_run": False
        } 