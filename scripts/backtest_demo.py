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
    
    # Calculate daily returns for each strategy
    daily_returns = {}
    for strategy, returns in results.items():
        # Ensure returns matches the date range
        strategy_returns = returns[:len(prices.index[lookback:])]
        # Convert returns to Series
        daily_returns[strategy] = pd.Series(strategy_returns, index=prices.index[lookback:][:len(strategy_returns)])
    
    # Plot basic and extended visualizations
    plot_backtest_results(performance, "Backtest Results")
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
    logger.info("Creating extended visualizations")
    
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
    
    # 6. Weight allocation visualization for each strategy (top strategies)
    top_strategies = metrics.sort_values('Sharpe Ratio', ascending=False).head(3).index
    
    for strategy in top_strategies:
        plt.figure(figsize=(14, 8))
        
        # Convert weights to DataFrame
        weights_df = pd.DataFrame()
        for period in weights_history[strategy]:
            date = period['date']
            weights = period['weights']
            weights_series = pd.Series(weights, name=date)
            weights_df = pd.concat([weights_df, weights_series.to_frame().T])
        
        # Select top assets by average weight
        if not weights_df.empty:
            top_assets = weights_df.mean().nlargest(8).index
            weights_df_top = weights_df[top_assets]
            
            # Plot as area chart
            ax = weights_df_top.plot(kind='area', stacked=True, colormap='viridis', alpha=0.7)
            
            plt.title(f'Weight Allocation Over Time - {strategy}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3)
            
            # Mark rebalancing dates
            rebalance_dates = weights_df.index
            for date in rebalance_dates:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"results/visualizations/weights_{strategy.replace(' ', '_')[:15]}.png", dpi=300)
            plt.close()
    
    # 7. Cumulative returns with rebalancing points
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
    plt.figure(figsize=(14, 8))
    
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
