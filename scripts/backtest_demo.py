#!/usr/bin/env python3
"""
Demonstration of various asset selection strategies and their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, List, Tuple, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.entities.portfolio import Portfolio
from src.domain.services.price_service import PriceService
from src.infrastructure.repositories.price_repository import PriceRepository
from src.infrastructure.data_sources.moex_api import MOEXDataSource
from src.infrastructure.data_sources.yahoo_api import YahooFinanceDataSource

# Set up logging
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler for backtest log
file_handler = logging.FileHandler('logs/backtest_demo.log', mode='w')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

async def run_backtest_demo():
    """Run a demonstration of backtest strategies."""
    logger.info("Starting backtest demonstration")
    
    # Create services
    price_repo = PriceRepository()
    # Initialize data sources
    data_sources = [YahooFinanceDataSource(), MOEXDataSource()]
    price_service = PriceService(repository=price_repo, data_sources=data_sources)
    
    # Try to load price data, or create sample data if not available
    try:
        tickers = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS", "MGNT", "AFLT"]
        prices_dict = await price_service.get_prices(tickers=tickers)
        
        # Convert dictionary of prices to DataFrame
        prices = pd.DataFrame()
        for ticker, price_list in prices_dict.items():
            if price_list:
                # Extract dates and close prices
                dates = [p.date for p in price_list]
                close_prices = [p.close or p.adjusted_close for p in price_list]
                ticker_df = pd.DataFrame({ticker: close_prices}, index=dates)
                prices = pd.concat([prices, ticker_df], axis=1)
        
        if prices.empty or len(prices) < 250:  # Ensure enough data points
            logger.info("Not enough price data found, creating sample data")
            prices = create_sample_price_data(tickers)
    except Exception as e:
        logger.warning(f"Error loading price data: {e}")
        logger.info("Creating sample price data")
        tickers = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS", "MGNT", "AFLT"]
        prices = create_sample_price_data(tickers)
    
    # Create sample factors
    factors = create_sample_factors(tickers)
    
    # Define portfolio strategies
    strategies = {
        "Industry Selection + Equal Weight": {
            "selector": Portfolio(factors, select_by="industry", select_metric="momentum"),
            "allocation": "equal"
        },
        "Industry Selection + Min Variance": {
            "selector": Portfolio(factors, select_by="industry", select_metric="momentum"),
            "allocation": "optimize",
            "target_metric": "variance"
        },
        "Optimize Sharpe + Equal Weight": {
            "selector": Portfolio(factors, select_by="optimize", select_metric="sharpe"),
            "allocation": "equal"
        },
        "Optimize Momentum + Min Variance": {
            "selector": Portfolio(factors, select_by="optimize", select_metric="momentum"),
            "allocation": "optimize",
            "target_metric": "variance"
        },
        "Optimize Momentum + Inverse Volatility": {
            "selector": Portfolio(factors, select_by="optimize", select_metric="momentum"),
            "allocation": "inverse_vol"
        },
        "Optimize Sharpe + Inverse CVaR": {
            "selector": Portfolio(factors, select_by="optimize", select_metric="sharpe"),
            "allocation": "cvar"
        },
        "Optimize CVaR + Max CVaR Ratio": {
            "selector": Portfolio(factors, select_by="optimize", select_metric="cvar"),
            "allocation": "optimize",
            "target_metric": "cvar_ratio"
        }
    }
    
    # Run backtests
    results = {}
    weights_history = {}
    
    logger.info("Running backtests for different strategies")
    
    # Define the backtest periods and parameters
    lookback = 252  # Use 1 year of data for selection
    cooling_period = 21  # Cooling period between training and testing (1 month)
    rebalance_freq = 21  # Monthly rebalancing (21 trading days)
    
    # Calculate approximate dates for reporting
    if len(prices) > 0:
        total_days = len(prices)
        train_start_idx = 0
        train_end_idx = int(total_days * 0.6)  # Use 60% for training
        cooling_end_idx = train_end_idx + cooling_period
        test_end_idx = total_days

        logger.info(f"Data split: Training ({prices.index[train_start_idx].date()} to "
                   f"{prices.index[train_end_idx-1].date()}), "
                   f"Cooling period ({prices.index[train_end_idx].date()} to "
                   f"{prices.index[cooling_end_idx-1].date() if cooling_end_idx < len(prices) else prices.index[-1].date()}), "
                   f"Testing ({prices.index[cooling_end_idx].date() if cooling_end_idx < len(prices) else 'N/A'} to "
                   f"{prices.index[test_end_idx-1].date() if test_end_idx < len(prices) else prices.index[-1].date()})")
    
    # Run in-sample backtest (training period) to set up initial strategies
    # This is only for strategy calibration, not for performance evaluation
    train_prices = prices.iloc[:int(len(prices) * 0.6)]
    logger.info(f"Running in-sample backtest with {len(train_prices)} data points")
    
    for name, strategy in strategies.items():
        logger.info(f"Running backtest for strategy: {name}")
        
        portfolio_returns = []
        strategy_weights = []
        
        # Setup portfolio
        selector = strategy["selector"]
        allocation_method = strategy["allocation"]
        target_metric = strategy.get("target_metric", "variance")
        
        # Skip points that don't have enough historical data
        start_idx = min(lookback, len(train_prices) - 1)
        
        # Run in-sample backtest for strategy calibration (training)
        for i in range(start_idx, len(train_prices), rebalance_freq):
            # Use historical data for selection and allocation
            historical_prices = train_prices.iloc[max(0, i-lookback):i]
            
            # Select assets
            selected_assets = selector.select(historical_prices)
            
            # Allocate weights
            weights = selector.allocate(
                historical_prices, 
                selected_assets, 
                method=allocation_method,
                target_metric=target_metric
            )
            
            # Store weights from training (for reference only)
            strategy_weights.append({
                'date': train_prices.index[i],
                'weights': weights.to_dict(),
                'period': 'training'
            })
    
    # Now run out-of-sample backtest (testing period) for performance evaluation
    # Skip the cooling period to prevent information leakage
    test_start_idx = int(len(prices) * 0.6) + cooling_period
    test_prices = prices.iloc[test_start_idx:]
    
    logger.info(f"Running out-of-sample backtest with {len(test_prices)} data points "
               f"after {cooling_period} days cooling period")
    
    # Check if we have enough data for testing
    if len(test_prices) <= lookback:
        logger.warning(f"Not enough testing data ({len(test_prices)} points) after cooling period. "
                      f"Need at least {lookback + 1} points for testing.")
        logger.info("Adjusting train-test split to ensure sufficient testing data")
        
        # Adjust the split to ensure we have enough testing data
        min_test_size = lookback + rebalance_freq * 2  # Enough for at least 2 rebalancing periods
        total_size = len(prices)
        
        if total_size < min_test_size + cooling_period + min_test_size:
            logger.warning("Dataset too small for proper train-test split with cooling period")
            
            # Use simpler approach: 80% train, 20% test without cooling
            test_start_idx = int(total_size * 0.8)
            test_prices = prices.iloc[test_start_idx:]
            
            # Adaptively reduce lookback period if the test dataset is still too small
            if len(test_prices) <= lookback:
                original_lookback = lookback
                # Reduce lookback to at most 75% of test data, but not less than 30 days
                lookback = max(min(int(len(test_prices) * 0.75), original_lookback), 30)
                logger.warning(f"Adaptively reducing lookback period from {original_lookback} to {lookback} days")
            
            logger.info(f"Using simplified split: training ({prices.index[0].date()} to "
                       f"{prices.index[test_start_idx-1].date()}), "
                       f"testing ({prices.index[test_start_idx].date()} to {prices.index[-1].date()})")
        else:
            # Adjust split to ensure minimum test size
            test_portion = (min_test_size + cooling_period) / total_size
            train_portion = 1 - test_portion
            
            test_start_idx = int(total_size * train_portion) + cooling_period
            test_prices = prices.iloc[test_start_idx:]
            
            # Adaptively reduce lookback period if the test dataset is still too small
            if len(test_prices) <= lookback:
                original_lookback = lookback
                # Reduce lookback to at most 75% of test data, but not less than 30 days
                lookback = max(min(int(len(test_prices) * 0.75), original_lookback), 30)
                logger.warning(f"Adaptively reducing lookback period from {original_lookback} to {lookback} days")
            
            logger.info(f"Adjusted split: training ({prices.index[0].date()} to "
                       f"{prices.index[int(total_size * train_portion)-1].date()}), "
                       f"cooling ({prices.index[int(total_size * train_portion)].date()} to "
                       f"{prices.index[test_start_idx-1].date()}), "
                       f"testing ({prices.index[test_start_idx].date()} to {prices.index[-1].date()})")
    
    for name, strategy in strategies.items():
        logger.info(f"Testing strategy: {name}")
        
        portfolio_returns = []
        portfolio_dates = []  # Track dates for each return
        
        # Setup portfolio
        selector = strategy["selector"]
        allocation_method = strategy["allocation"]
        target_metric = strategy.get("target_metric", "variance")
        
        # Skip points that don't have enough historical data
        start_idx = min(lookback, len(test_prices) - 1)
        
        # Check if we can run at least one test
        if start_idx >= len(test_prices) - 1:
            logger.warning(f"Test dataset too small for strategy: {name}. Skipping.")
            continue
        
        # Calculate rebalancing dates to ensure consistent frequency
        # First rebalancing date is after we have enough historical data
        first_rebalance_date = test_prices.index[start_idx]
        
        # Generate all subsequent rebalancing dates at exact intervals
        rebalance_dates = []
        current_date = first_rebalance_date
        while current_date <= test_prices.index[-1]:
            rebalance_dates.append(current_date)
            # Find the date that's exactly rebalance_freq days later (or closest available)
            target_date = current_date + pd.Timedelta(days=rebalance_freq)
            # Find the closest available date in our dataset
            if target_date > test_prices.index[-1]:
                break
            date_idx = test_prices.index.searchsorted(target_date)
            if date_idx >= len(test_prices.index):
                break
            current_date = test_prices.index[date_idx]
        
        logger.info(f"Strategy {name}: Scheduled {len(rebalance_dates)} rebalancing events")
            
        # Run out-of-sample backtest for performance evaluation (testing)
        last_weights = None
        for rebalance_date in rebalance_dates:
            # Find the index of this rebalance date
            date_idx = test_prices.index.get_loc(rebalance_date)
            
            # Get historical data up to this rebalance date
            current_date_idx = test_start_idx + date_idx
            historical_prices = prices.iloc[max(0, current_date_idx-lookback):current_date_idx]
            
            # Check if we have enough historical data
            if len(historical_prices) < min(30, lookback / 2):  # At least 30 days or half the lookback
                logger.warning(f"Insufficient historical data at {rebalance_date} for {name}. Using previous weights.")
                if last_weights is None:
                    # If no previous weights, use equal weights
                    selected_assets = list(test_prices.columns[:min(5, len(test_prices.columns))])
                    weights = pd.Series({asset: 1.0 / len(selected_assets) for asset in selected_assets})
                else:
                    # Keep using previous weights
                    weights = last_weights
            else:
                # Select assets
                selected_assets = selector.select(historical_prices)
                
                # Allocate weights
                weights = selector.allocate(
                    historical_prices, 
                    selected_assets, 
                    method=allocation_method,
                    target_metric=target_metric
                )
            
            # Store current weights for next iteration
            last_weights = weights
            
            # Store weights from testing
            strategy_weights.append({
                'date': rebalance_date,
                'weights': weights.to_dict(),
                'period': 'testing'
            })
            
            # Calculate forward returns until next rebalancing
            next_idx = date_idx + 1
            while next_idx < len(test_prices.index):
                if next_idx >= len(test_prices.index) or (len(rebalance_dates) > rebalance_dates.index(rebalance_date) + 1 and 
                                                   test_prices.index[next_idx] >= rebalance_dates[rebalance_dates.index(rebalance_date) + 1]):
                    break
                    
                # Calculate daily return
                if next_idx > 0:  # Skip first day as we can't calculate return
                    day_return = 0
                    for asset, weight in weights.items():
                        if asset in test_prices.columns:
                            # Calculate daily return for this asset
                            asset_return = (test_prices[asset].iloc[next_idx] / test_prices[asset].iloc[next_idx-1]) - 1
                            day_return += weight * asset_return
                    
                    # Store the return and its date
                    portfolio_returns.append(day_return)
                    portfolio_dates.append(test_prices.index[next_idx])
                
                next_idx += 1
        
        # Create a complete series with all dates in test period
        full_returns = pd.Series(0.0, index=test_prices.index)
        
        # Fill in our actual returns at the corresponding dates
        for date, ret in zip(portfolio_dates, portfolio_returns):
            full_returns[date] = ret
        
        # Store results (only from testing period)
        results[name] = full_returns.tolist()
        weights_history[name] = strategy_weights
    
    # Check if we have any results from testing period
    if all(len(returns) == 0 for returns in results.values()):
        logger.warning("No results generated from testing period. Test dataset may be too small.")
        # Create empty performance dataframe with the right index
        if len(test_prices) > 0:
            performance = pd.DataFrame(index=test_prices.index)
        else:
            performance = pd.DataFrame()
    else:
        # Create a complete series of dates for the testing period
        # This ensures we have a continuous time series even if strategies skip some days
        all_test_dates = test_prices.index
        
        # Calculate performance metrics (only on testing data)
        performance = calculate_portfolio_performance(results, all_test_dates)
    
    # Calculate daily returns for each strategy with proper date alignment
    daily_returns = {}
    
    for strategy, returns in results.items():
        if returns:  # Check if returns list is not empty
            # Create properly aligned series
            daily_returns[strategy] = pd.Series(returns, index=test_prices.index)
        else:
            # Create empty series with the right index
            daily_returns[strategy] = pd.Series(index=test_prices.index)
    
    # Log summary statistics for verification
    logger.info("Performance summary:")
    for strategy in results.keys():
        if not performance.empty and strategy in performance.columns:
            final_return = performance[strategy].iloc[-1] * 100 if len(performance) > 0 else 0
            logger.info(f"  {strategy}: {final_return:.2f}% return, "
                       f"{len([r for r in results[strategy] if r != 0])} active trading days")
    
    # Plot basic and extended visualizations
    if not performance.empty:
        plot_backtest_results(performance, "Out-of-Sample Backtest Results")
        plot_extended_visualizations(
            performance=performance, 
            daily_returns=daily_returns,
            weights_history=weights_history,
            lookback=lookback,
            rebalance_freq=rebalance_freq
        )
        
        # Export to Excel and HTML report
        export_to_excel(prices, factors, results, performance, weights_history)
        create_html_report(performance, daily_returns, weights_history)
    else:
        logger.warning("No performance data generated. Skipping visualizations and reports.")
    
    logger.info("Backtest demonstration completed")
    
    return performance

def create_sample_price_data(tickers: List[str]) -> pd.DataFrame:
    """
    Create sample price data for backtesting.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with daily prices
    """
    # Create a date range for the past 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Create business day date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initialize price dataframe
    prices = pd.DataFrame(index=date_range)
    
    # Generate random walk prices for each ticker
    np.random.seed(42)  # For reproducibility
    
    for ticker in tickers:
        # Initial price between 100 and 1000
        initial_price = np.random.uniform(100, 1000)
        
        # Daily returns with drift
        drift = np.random.uniform(0.0001, 0.0005)  # Small positive drift
        volatility = np.random.uniform(0.01, 0.02)  # Daily volatility
        
        # Generate log returns
        log_returns = np.random.normal(drift, volatility, len(date_range))
        
        # Convert to price series
        price_series = initial_price * np.exp(np.cumsum(log_returns))
        
        # Add to dataframe
        prices[ticker] = price_series
    
    return prices

def create_sample_factors(tickers: List[str]) -> pd.DataFrame:
    """
    Create sample factor data for asset selection.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with factors for each ticker
    """
    # Create industries and sectors
    industries = ["Banking", "Oil & Gas", "Telecom", "Mining", "Retail"]
    sectors = ["Financials", "Energy", "Communication", "Materials", "Consumer"]
    
    # Create factor dataframe
    factors = pd.DataFrame(index=tickers)
    
    # Assign random industries and sectors
    np.random.seed(42)  # For reproducibility
    
    factors["industry"] = np.random.choice(industries, size=len(tickers))
    factors["sector"] = np.random.choice(sectors, size=len(tickers))
    
    # Add size factor (market cap in billions)
    factors["size"] = np.random.uniform(1, 100, size=len(tickers))
    
    # Add value factor (P/E ratio)
    factors["value"] = np.random.uniform(5, 30, size=len(tickers))
    
    return factors

def calculate_portfolio_performance(
    results: Dict[str, List[float]], 
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Calculate performance metrics for each portfolio strategy.
    
    Args:
        results: Dictionary with strategy name and returns
        dates: Date index for the returns
        
    Returns:
        DataFrame with performance metrics
    """
    performance = pd.DataFrame(index=dates)
    
    for strategy, returns in results.items():
        # Handle case where returns might be shorter than dates
        strategy_returns = returns[:len(dates)]
        
        # Ensure we have a return for each date by padding with zeros if needed
        if len(strategy_returns) < len(dates):
            logger.warning(f"Strategy {strategy} has {len(strategy_returns)} returns but expected {len(dates)}. "
                           f"Padding with zeros.")
            # Pad with zeros to match length
            strategy_returns.extend([0.0] * (len(dates) - len(strategy_returns)))
        
        # Convert returns to Series
        returns_series = pd.Series(strategy_returns, index=dates[:len(strategy_returns)])
        
        # Fill any missing dates with zero returns to avoid discontinuities
        if returns_series.isna().any():
            logger.warning(f"Strategy {strategy} has {returns_series.isna().sum()} missing values. Filling with zeros.")
            returns_series = returns_series.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_series).cumprod() - 1
        
        # Add to performance dataframe
        performance[strategy] = cumulative_returns
    
    return performance

def plot_backtest_results(performance: pd.DataFrame, title: str) -> None:
    """
    Plot backtest results for different strategies.
    
    Args:
        performance: DataFrame with strategy performance
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for strategy in performance.columns:
        plt.plot(performance.index, performance[strategy] * 100, label=strategy)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/backtest_results.png")
    plt.close()

def plot_extended_visualizations(
    performance: pd.DataFrame,
    daily_returns: Dict[str, pd.Series],
    weights_history: Dict[str, List[Dict]],
    lookback: int,
    rebalance_freq: int
) -> None:
    """
    Create extended visualizations for backtest results.
    
    Args:
        performance: DataFrame with strategy performance
        daily_returns: Dictionary mapping strategy names to daily return series
        weights_history: Dictionary mapping strategy names to weights history
        lookback: Lookback period for asset selection
        rebalance_freq: Rebalancing frequency in days
    """
    # Check if performance data is empty
    if performance.empty or len(performance) == 0:
        logger.warning("Performance data is empty, skipping extended visualizations")
        return
    
    # Set up the visualization style
    sns.set(style="whitegrid")
    os.makedirs("results/visualizations", exist_ok=True)
    
    # 1. Drawdown visualization for each strategy
    plt.figure(figsize=(14, 8))
    
    for strategy in performance.columns:
        # Calculate drawdowns
        wealth_index = 1 + performance[strategy]
        running_max = wealth_index.cummax()
        drawdowns = (wealth_index / running_max - 1) * 100
        
        plt.plot(drawdowns.index, drawdowns, label=strategy, linewidth=2)
    
    plt.title('Drawdowns for Different Strategies', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/visualizations/drawdowns.png", dpi=300)
    plt.close()
    
    # 2. Distribution of daily returns
    plt.figure(figsize=(14, 8))
    
    for strategy, returns in daily_returns.items():
        sns.kdeplot(returns * 100, label=strategy, fill=True, alpha=0.2)
    
    plt.title('Distribution of Daily Returns', fontsize=16)
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/visualizations/return_distribution.png", dpi=300)
    plt.close()
    
    # 3. Rolling volatility (30-day window)
    plt.figure(figsize=(14, 8))
    
    for strategy, returns in daily_returns.items():
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        plt.plot(rolling_vol.index, rolling_vol, label=strategy, linewidth=2)
    
    plt.title('Rolling 30-Day Annualized Volatility', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility (%)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/visualizations/rolling_volatility.png", dpi=300)
    plt.close()
    
    # 4. Rolling Sharpe ratio (90-day window)
    plt.figure(figsize=(14, 8))
    
    for strategy, returns in daily_returns.items():
        rolling_return = returns.rolling(90).mean() * 252
        rolling_vol = returns.rolling(90).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        plt.plot(rolling_sharpe.index, rolling_sharpe, label=strategy, linewidth=2)
    
    plt.title('Rolling 90-Day Sharpe Ratio', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/visualizations/rolling_sharpe.png", dpi=300)
    plt.close()
    
    # 5. Performance comparison as a heatmap
    # Calculate key metrics for each strategy
    metrics = pd.DataFrame(index=performance.columns)
    
    # Check if the performance DataFrame has data
    if len(performance) > 0:
        # Total return
        metrics['Total Return (%)'] = performance.iloc[-1] * 100
        
        # Annualized return
        days = len(performance)
        years = days / 252
        metrics['Annualized Return (%)'] = ((1 + performance.iloc[-1]) ** (1 / years) - 1) * 100
        
        # Annualized volatility
        metrics['Annualized Volatility (%)'] = [daily_returns[s].std() * np.sqrt(252) * 100 for s in performance.columns]
        
        # Sharpe ratio
        metrics['Sharpe Ratio'] = metrics['Annualized Return (%)'] / metrics['Annualized Volatility (%)']
        
        # Maximum drawdown
        metrics['Maximum Drawdown (%)'] = [calculate_max_drawdown(performance[s]) * 100 for s in performance.columns]
        
        # Calmar ratio
        metrics['Calmar Ratio'] = -metrics['Annualized Return (%)'] / metrics['Maximum Drawdown (%)']
        
        # Sortino ratio (downside deviation)
        sortino_values = []
        for strategy in performance.columns:
            neg_returns = daily_returns[strategy].copy()
            neg_returns[neg_returns > 0] = 0
            downside_dev = neg_returns.std() * np.sqrt(252)
            sortino = metrics.loc[strategy, 'Annualized Return (%)'] / 100 / downside_dev if downside_dev > 0 else 0
            sortino_values.append(sortino)
        metrics['Sortino Ratio'] = sortino_values
        
        # Plot heatmap of metrics
        plt.figure(figsize=(16, 10))
        metrics_heatmap = metrics.copy()
        
        # Standardize values for better visualization
        for col in metrics_heatmap.columns:
            metrics_heatmap[col] = (metrics_heatmap[col] - metrics_heatmap[col].mean()) / metrics_heatmap[col].std()
        
        sns.heatmap(metrics_heatmap.T, annot=metrics.T, fmt='.2f', cmap='RdYlGn', 
                    linewidths=.5, center=0, cbar_kws={"shrink": .8})
        
        plt.title('Strategy Performance Metrics Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig("results/visualizations/performance_heatmap.png", dpi=300)
        plt.close()
    else:
        logger.warning("Not enough performance data to calculate metrics")
    
    # 6. Weight allocation visualization for each strategy (top strategies)
    # Only proceed if we have performance data and metrics calculated
    if not metrics.empty and 'Sharpe Ratio' in metrics.columns:
        top_strategies = metrics.sort_values('Sharpe Ratio', ascending=False).head(3).index
        
        for strategy in top_strategies:
            # Create a new figure
            plt.figure(figsize=(14, 8))
            
            # Collect weights data for testing period
            strategy_weights = []
            strategy_dates = []
            
            for period in weights_history[strategy]:
                if period['period'] == 'testing':  # Only use testing period weights
                    strategy_weights.append(period['weights'])
                    strategy_dates.append(period['date'])
            
            # Skip if we don't have enough data
            if len(strategy_weights) < 2:
                logger.warning(f"Not enough rebalancing events for strategy {strategy}, skipping visualization")
                continue
                
            # Get last date from performance dataframe for visualization end
            last_date = performance.index[-1]
            
            # Select top assets (limit to 8 for readability)
            all_assets = set()
            for w in strategy_weights:
                all_assets.update(w.keys())
            
            # Calculate average weight for each asset
            avg_weights = {}
            for asset in all_assets:
                weights = [w.get(asset, 0) for w in strategy_weights]
                avg_weights[asset] = sum(weights) / len(weights)
            
            # Select top assets
            top_assets = sorted(avg_weights.keys(), key=lambda k: avg_weights[k], reverse=True)[:8]
            
            # Create basic step plots for each asset
            bottom = np.zeros(len(strategy_dates) + 1)  # +1 for the last point
            
            # Add final date for complete visualization
            plot_dates = strategy_dates + [last_date]
            
            # Use viridis colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_assets)))
            
            for i, asset in enumerate(top_assets):
                # Get weights for this asset at each rebalancing date
                asset_weights = [w.get(asset, 0) for w in strategy_weights]
                
                # Add the last weight again for step visualization
                asset_weights.append(asset_weights[-1])
                
                # Plot as a step function
                plt.fill_between(
                    plot_dates,
                    bottom,
                    bottom + asset_weights, 
                    step='post',  # Use post-step for accurate visualization
                    alpha=0.7,
                    label=asset,
                    color=colors[i]
                )
                
                # Update bottom for stacking
                bottom = bottom + asset_weights
            
            # Add labels and title
            plt.title(f'Weight Allocation Over Time - {strategy}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.ylim(0, 1.05)  # Set y-axis limits
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3)
            
            # Mark rebalancing dates with vertical lines
            for date in strategy_dates:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            
            plt.tight_layout()
            plt.savefig(f"results/visualizations/weights_{strategy.replace(' ', '_')[:15]}.png", dpi=300)
            plt.close()
    else:
        logger.warning("Skipping weight allocation visualization due to insufficient performance data")
    
    # 7. Cumulative returns with rebalancing points
    if not performance.empty:
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative returns
        for strategy in performance.columns:
            plt.plot(performance.index, performance[strategy] * 100, label=strategy, linewidth=2)
        
        # Mark rebalancing dates for the first strategy (they are the same for all)
        if weights_history and weights_history[list(weights_history.keys())[0]]:
            rebalance_dates = [period['date'] for period in weights_history[list(weights_history.keys())[0]]]
            for date in rebalance_dates:
                plt.axvline(x=date, color='gray', linestyle='--', alpha=0.5)
        
        plt.title('Cumulative Returns with Rebalancing Points', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/visualizations/returns_with_rebalancing.png", dpi=300)
        plt.close()
    
    # 8. Monthly returns heatmap for the best strategy
    # Only proceed if we have performance data and metrics calculated
    if not metrics.empty and 'Sharpe Ratio' in metrics.columns:
        best_strategy = metrics.sort_values('Sharpe Ratio', ascending=False).index[0]
        
        # Create monthly returns
        try:
            # Try to resample to month end
            monthly_returns = daily_returns[best_strategy].resample('M').mean() * 21 * 100  # Approximate monthly return
            
            # Create pivot table for heatmap
            monthly_returns_pivot = pd.pivot_table(
                monthly_returns.reset_index(),
                values=0,
                index=[monthly_returns.index.year],
                columns=[monthly_returns.index.month],
                aggfunc='sum'
            )
            monthly_returns_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                        linewidths=.5, center=0, cbar_kws={"shrink": .8})
            
            plt.title(f'Monthly Returns (%) - {best_strategy}', fontsize=16)
            plt.tight_layout()
            plt.savefig("results/visualizations/monthly_returns_heatmap.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create monthly returns heatmap: {e}")
            
            # Alternative visualization: returns distribution by month
            plt.figure(figsize=(14, 8))
            
            # Group returns by month
            returns_by_month = {}
            for i, ret in daily_returns[best_strategy].items():
                month = i.month
                if month not in returns_by_month:
                    returns_by_month[month] = []
                returns_by_month[month].append(ret * 100)
            
            # Create boxplot
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            data_to_plot = [returns_by_month.get(m+1, []) for m in range(12)]
            
            plt.boxplot(data_to_plot, labels=month_names)
            plt.title(f'Daily Returns Distribution by Month - {best_strategy}', fontsize=16)
            plt.ylabel('Daily Return (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/visualizations/returns_by_month.png", dpi=300)
            plt.close()
        
        # Create combined plot of top 3 strategies
        if len(metrics) >= 3:  # Only if we have at least 3 strategies
            plt.figure(figsize=(14, 8))
            
            top_strategies = metrics.sort_values('Sharpe Ratio', ascending=False).head(3).index
            for strategy in top_strategies:
                plt.plot(performance.index, performance[strategy] * 100, label=strategy, linewidth=2)
            
            plt.title('Top 3 Strategies by Sharpe Ratio', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return (%)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/visualizations/top_strategies.png", dpi=300)
            plt.close()
    
    logger.info(f"Extended visualizations saved to results/visualizations/")

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown for a series of returns.
    
    Args:
        returns: Series of cumulative returns
        
    Returns:
        Maximum drawdown as a decimal
    """
    # Add 1 to get a price series from returns
    wealth_index = 1 + returns
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdowns
    drawdowns = wealth_index / previous_peaks - 1
    
    # Get the minimum drawdown
    return drawdowns.min()

def export_to_excel(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    results: Dict[str, List[float]],
    performance: pd.DataFrame,
    weights_history: Dict[str, List[Dict]]
) -> None:
    """
    Export backtest results to Excel.
    
    Args:
        prices: DataFrame with price data
        factors: DataFrame with factor data
        results: Dictionary with strategy returns
        performance: DataFrame with strategy performance
        weights_history: Dictionary with portfolio weights over time
    """
    # Check if performance data is empty
    if performance.empty:
        logger.warning("Performance data is empty, skipping Excel export")
        return
        
    # Create Excel writer
    os.makedirs("results", exist_ok=True)
    writer = pd.ExcelWriter("results/backtest_results.xlsx", engine="xlsxwriter")
    
    # Write price data
    prices.to_excel(writer, sheet_name="Prices")
    
    # Write factor data
    factors.to_excel(writer, sheet_name="Factors")
    
    # Write performance data
    performance.to_excel(writer, sheet_name="Performance")
    
    # Write weights history for each strategy
    for strategy, weights in weights_history.items():
        # Create a dataframe for weights
        weights_df = pd.DataFrame()
        
        for period in weights:
            date = period['date']
            period_weights = period['weights']
            
            # Convert weights to Series
            weights_series = pd.Series(period_weights, name=date)
            
            # Add to dataframe
            weights_df = pd.concat([weights_df, weights_series.to_frame().T])
        
        # Write to Excel
        # Shorten strategy name to make valid sheet name (31 chars max)
        sheet_name = f"{strategy[:15]}_Weights"  # More aggressively limit sheet name length
        weights_df.to_excel(writer, sheet_name=sheet_name)
    
    # Write summary statistics
    summary = pd.DataFrame(index=performance.columns)
    
    # Check if we have enough data for statistics
    if len(performance) > 0:
        # Calculate annualized return
        days = len(performance)
        years = days / 252
        summary["Annualized Return (%)"] = ((1 + performance.iloc[-1]) ** (1 / years) - 1) * 100
        
        # Calculate volatility
        returns = performance.pct_change().dropna()
        summary["Annualized Volatility (%)"] = returns.std() * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio
        summary["Sharpe Ratio"] = summary["Annualized Return (%)"] / summary["Annualized Volatility (%)"]
        
        # Calculate maximum drawdown
        for strategy in performance.columns:
            cumulative = 1 + performance[strategy]
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            summary.loc[strategy, "Maximum Drawdown (%)"] = drawdown.min() * 100
        
        # Calculate Calmar ratio
        summary["Calmar Ratio"] = -summary["Annualized Return (%)"] / summary["Maximum Drawdown (%)"]
    else:
        logger.warning("Not enough performance data to calculate summary statistics")
    
    # Write to Excel
    summary.to_excel(writer, sheet_name="Summary")
    
    # Save the Excel file
    writer.close()
    
    logger.info(f"Results exported to results/backtest_results.xlsx")

def create_html_report(
    performance: pd.DataFrame,
    daily_returns: Dict[str, pd.Series],
    weights_history: Dict[str, List[Dict]]
) -> None:
    """
    Create an HTML report with visualizations.
    
    Args:
        performance: DataFrame with strategy performance
        daily_returns: Dictionary mapping strategy names to daily return series
        weights_history: Dictionary mapping strategy names to weights history
    """
    # Check if performance data is empty
    if performance.empty or len(performance) == 0:
        logger.warning("Performance data is empty, skipping HTML report generation")
        return
        
    logger.info("Creating HTML report")
    
    # Calculate metrics for the report
    metrics = pd.DataFrame(index=performance.columns)
    
    # Total return
    metrics['Total Return (%)'] = performance.iloc[-1] * 100
    
    # Annualized return
    days = len(performance)
    years = days / 252
    metrics['Annual Return (%)'] = ((1 + performance.iloc[-1]) ** (1 / years) - 1) * 100
    
    # Annualized volatility
    metrics['Annual Vol (%)'] = [daily_returns[s].std() * np.sqrt(252) * 100 for s in performance.columns]
    
    # Sharpe ratio
    metrics['Sharpe Ratio'] = metrics['Annual Return (%)'] / metrics['Annual Vol (%)']
    
    # Maximum drawdown
    metrics['Max Drawdown (%)'] = [calculate_max_drawdown(performance[s]) * 100 for s in performance.columns]
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Results Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; background-color: #f5f5f5; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        .metrics-table th {{ background-color: #f2f2f2; text-align: center; }}
        .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metrics-table tr:hover {{ background-color: #f1f1f1; }}
        .visualization {{ margin: 30px 0; text-align: center; }}
        .visualization img {{ max-width: 100%; height: auto; box-shadow: 0 0 5px rgba(0,0,0,0.2); }}
        .section {{ margin: 40px 0; }}
        .highlight {{ background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #777; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Results Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <p>Comparison of different asset selection and allocation strategies</p>
            <div class="highlight">
                <p>Best strategy: <strong>{metrics.sort_values('Sharpe Ratio', ascending=False).index[0]}</strong> with a Sharpe ratio of {metrics.sort_values('Sharpe Ratio', ascending=False)['Sharpe Ratio'].iloc[0]:.2f}</p>
            </div>
            {metrics.round(2).to_html(classes='metrics-table')}
        </div>
        
        <div class="section">
            <h2>Performance Visualization</h2>
            <div class="visualization">
                <h3>Cumulative Returns</h3>
                <img src="visualizations/returns_with_rebalancing.png" alt="Cumulative Returns">
            </div>
            
            <div class="visualization">
                <h3>Top 3 Strategies by Sharpe Ratio</h3>
                <img src="visualizations/top_strategies.png" alt="Top Strategies">
            </div>
            
            <div class="visualization">
                <h3>Drawdowns</h3>
                <img src="visualizations/drawdowns.png" alt="Drawdowns">
            </div>
        </div>
        
        <div class="section">
            <h2>Risk and Return Analysis</h2>
            <div class="visualization">
                <h3>Return Distribution</h3>
                <img src="visualizations/return_distribution.png" alt="Return Distribution">
            </div>
            
            <div class="visualization">
                <h3>Rolling Volatility</h3>
                <img src="visualizations/rolling_volatility.png" alt="Rolling Volatility">
            </div>
            
            <div class="visualization">
                <h3>Rolling Sharpe Ratio</h3>
                <img src="visualizations/rolling_sharpe.png" alt="Rolling Sharpe Ratio">
            </div>
            
            <div class="visualization">
                <h3>Performance Metrics Heatmap</h3>
                <img src="visualizations/performance_heatmap.png" alt="Performance Heatmap">
            </div>
            
            <div class="visualization">
                <h3>Monthly Returns</h3>
                <img src="visualizations/monthly_returns_heatmap.png" alt="Monthly Returns Heatmap">
            </div>
        </div>
        
        <div class="section">
            <h2>Portfolio Allocation</h2>
            <p>Asset weights over time for top strategies:</p>
"""
    
    # Add weight images HTML for top strategies
    for strategy in metrics.sort_values('Sharpe Ratio', ascending=False).head(3).index:
        safe_name = strategy.replace(' ', '_')[:15]
        html_content += f"""
            <div class="visualization">
                <h3>Weights: {strategy}</h3>
                <img src="visualizations/weights_{safe_name}.png" alt="Weight Allocation">
            </div>
"""
    
    # Add footer
    html_content += """
        </div>
        
        <div class="footer">
            <p>Generated by Asset Selection Backtest Demo. Data is simulated for demonstration purposes.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_report.html", "w") as f:
        f.write(html_content)
    
    logger.info("HTML report created at results/backtest_report.html")

def main():
    """Main entry point of the script."""
    asyncio.run(run_backtest_demo())

if __name__ == "__main__":
    main()
