#!/usr/bin/env python3
"""
Demonstration of various asset selection strategies and their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_backtest_demo():
    """Run a demonstration of backtest strategies."""
    logger.info("Starting backtest demonstration")
    
    # Create services
    price_repo = PriceRepository()
    # Initialize data sources
    data_sources = [YahooFinanceDataSource(), MOEXDataSource()]
    price_service = PriceService(price_repo, data_sources)
    
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
        }
    }
    
    # Run backtests
    results = {}
    weights_history = {}
    
    logger.info("Running backtests for different strategies")
    
    # Define the backtest period
    lookback = 252  # Use 1 year of data for selection
    rebalance_freq = 21  # Monthly rebalancing (21 trading days)
    
    for name, strategy in strategies.items():
        logger.info(f"Running backtest for strategy: {name}")
        
        portfolio_returns = []
        strategy_weights = []
        
        # Setup portfolio
        selector = strategy["selector"]
        allocation_method = strategy["allocation"]
        target_metric = strategy.get("target_metric", "variance")
        
        # Run backtest over time
        for i in range(lookback, len(prices), rebalance_freq):
            # Use historical data for selection and allocation
            historical_prices = prices.iloc[i-lookback:i]
            
            # Select assets
            selected_assets = selector.select(historical_prices)
            
            # Allocate weights
            weights = selector.allocate(
                historical_prices, 
                selected_assets, 
                method=allocation_method,
                target_metric=target_metric
            )
            
            # Store weights
            strategy_weights.append({
                'date': prices.index[i],
                'weights': weights.to_dict()
            })
            
            # Calculate forward returns
            if i + rebalance_freq <= len(prices):
                forward_prices = prices.iloc[i:i+rebalance_freq]
                forward_returns = forward_prices.pct_change().fillna(0)
                
                # Calculate portfolio return
                daily_returns = []
                for day in range(len(forward_returns)):
                    # Get daily return for portfolio
                    day_return = 0
                    for asset, weight in weights.items():
                        if asset in forward_returns.columns:
                            day_return += weight * forward_returns.iloc[day][asset]
                    daily_returns.append(day_return)
                
                portfolio_returns.extend(daily_returns)
        
        # Store results
        results[name] = portfolio_returns
        weights_history[name] = strategy_weights
    
    # Calculate performance metrics
    performance = calculate_portfolio_performance(results, prices.index[lookback:])
    
    # Plot results
    plot_backtest_results(performance, "Backtest Results")
    
    # Export to Excel
    export_to_excel(prices, factors, results, performance, weights_history)
    
    logger.info("Backtest demonstration completed successfully")
    
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
    performance = pd.DataFrame()
    
    for strategy, returns in results.items():
        # Ensure returns matches the date range
        strategy_returns = returns[:len(dates)]
        
        # Convert returns to Series
        returns_series = pd.Series(strategy_returns, index=dates[:len(strategy_returns)])
        
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
    
    # Write to Excel
    summary.to_excel(writer, sheet_name="Summary")
    
    # Save the Excel file
    writer.close()
    
    logger.info(f"Results exported to results/backtest_results.xlsx")

def main():
    """Main entry point of the script."""
    asyncio.run(run_backtest_demo())

if __name__ == "__main__":
    main()
