#!/usr/bin/env python3
"""
Demonstration of portfolio rebalancing.
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

async def run_rebalance_demo():
    """Run a demonstration of portfolio rebalancing."""
    logger.info("Starting rebalance demonstration")
    
    # Create services
    price_repo = PriceRepository()
    # Initialize data sources
    data_sources = [YahooFinanceDataSource(), MOEXDataSource()]
    price_service = PriceService(price_repo, data_sources)
    
    # Define a list of tickers to work with
    tickers = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS", "MGNT", "AFLT"]
    
    # Get price data or create sample data
    try:
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
        
        if prices.empty or len(prices) < 30:  # Ensure enough data points
            logger.info("Not enough price data found, creating sample data")
            prices = create_sample_price_data(tickers)
    except Exception as e:
        logger.warning(f"Error loading price data: {e}")
        logger.info("Creating sample price data")
        prices = create_sample_price_data(tickers)
    
    # Create sample factors
    factors = create_sample_factors(tickers)
    
    # Create a portfolio with an asset selection and allocation strategy
    portfolio = Portfolio(factors, select_by="optimize", select_metric="sharpe")
    
    # Create a hypothetical current portfolio
    current_portfolio = create_current_portfolio(prices, tickers[:6], 1000000)
    
    # Display current portfolio
    logger.info("\n=== Current Portfolio ===")
    logger.info(f"Total value: {current_portfolio['market_value'].sum():.2f}")
    logger.info(f"Number of positions: {len(current_portfolio)}")
    
    for ticker, row in current_portfolio.iterrows():
        logger.info(f"{ticker}: {row['quantity']} shares, {row['market_value']:.2f} value, {row['weight']*100:.2f}% weight")
    
    # Calculate target weights
    selected_assets = portfolio.select(prices)
    target_weights = portfolio.allocate(prices, selected_assets, method="optimize", target_metric="sharpe")
    
    # Display target portfolio
    logger.info("\n=== Target Portfolio ===")
    for ticker, weight in target_weights.items():
        logger.info(f"{ticker}: {weight*100:.2f}% weight")
    
    # Add additional information for rebalancing
    additional_info = pd.DataFrame(index=tickers)
    additional_info['lot'] = 1  # Assume lot size of 1 for simplicity
    additional_info['ticker'] = tickers
    
    # Calculate rebalance orders with different thresholds
    thresholds = [0.05, 0.01]
    
    for threshold in thresholds:
        logger.info(f"\n=== Rebalance Orders (Threshold: {threshold*100}%) ===")
        
        # Calculate rebalance orders
        orders, _ = portfolio.calculate_rebalance_orders(
            current_portfolio,
            prices,
            cash=50000,  # Assume 50,000 available cash
            method="optimize",
            target_metric="sharpe",
            threshold=threshold
        )
        
        # Display rebalance orders
        logger.info(f"Generated {len(orders)} orders")
        
        for ticker, order in orders.iterrows():
            action = "BUY" if order['is_buy'] else "SELL"
            logger.info(f"{action} {ticker}: {abs(int(order['order_quantity']))} shares, {abs(order['order_value']):.2f} value")
        
        # Calculate MAPE between current and target weights
        current_weights = pd.Series(current_portfolio['weight'].to_dict())
        mape = portfolio.get_mean_absolute_percentage_error(current_weights, target_weights)
        logger.info(f"Mean Absolute Percentage Error: {mape*100:.2f}%")
        
        # Simulate execution (dry run)
        execution_results = portfolio.execute_rebalance(orders, dry_run=True)
        logger.info(f"Total rebalance value: {execution_results['total_value']:.2f}")
    
    # Alternative approach: prepare positions directly
    logger.info("\n=== Prepare Positions Approach ===")
    prepared_positions = portfolio.prepare_rebalance_positions(
        current_portfolio,
        target_weights,
        additional_info
    )
    
    logger.info(f"Generated {len(prepared_positions)} position changes")
    for ticker, position in prepared_positions.iterrows():
        action = "BUY" if position['delta_quantity_for_order'] > 0 else "SELL"
        logger.info(f"{action} {ticker}: {abs(int(position['delta_quantity_for_order']))} shares, {abs(position['delta_market_value']):.2f} value")
    
    # Display projected portfolio after rebalancing
    logger.info("\n=== Projected Portfolio After Rebalancing ===")
    
    # Calculate projected portfolio
    projected_portfolio = current_portfolio.copy()
    for ticker, order in orders.iterrows():
        if ticker in projected_portfolio.index:
            projected_portfolio.loc[ticker, 'quantity'] += order['order_quantity']
            projected_portfolio.loc[ticker, 'market_value'] = projected_portfolio.loc[ticker, 'quantity'] * projected_portfolio.loc[ticker, 'current_price']
        else:
            # New position
            projected_portfolio.loc[ticker] = {
                'quantity': order['order_quantity'],
                'current_price': order['current_price'],
                'market_value': order['order_quantity'] * order['current_price'],
                'lot': 1,
                'weight': 0  # Will be recalculated
            }
    
    # Recalculate weights
    total_value = projected_portfolio['market_value'].sum()
    projected_portfolio['weight'] = projected_portfolio['market_value'] / total_value
    
    logger.info(f"Projected total value: {total_value:.2f}")
    
    for ticker, row in projected_portfolio.iterrows():
        logger.info(f"{ticker}: {row['quantity']} shares, {row['market_value']:.2f} value, {row['weight']*100:.2f}% weight")
    
    # Calculate MAPE for projected portfolio
    projected_weights = pd.Series(projected_portfolio['weight'].to_dict())
    projected_mape = portfolio.get_mean_absolute_percentage_error(projected_weights, target_weights)
    logger.info(f"Projected Mean Absolute Percentage Error: {projected_mape*100:.2f}%")
    
    logger.info("Rebalance demonstration completed successfully")
    
    return {
        "current_portfolio": current_portfolio,
        "target_weights": target_weights,
        "orders": orders,
        "projected_portfolio": projected_portfolio
    }

def create_sample_price_data(tickers: List[str]) -> pd.DataFrame:
    """
    Create sample price data for demonstrations.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with daily prices
    """
    # Create a date range for the past 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
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

def create_current_portfolio(prices: pd.DataFrame, holdings: List[str], total_value: float = 1000000) -> pd.DataFrame:
    """
    Create a hypothetical current portfolio.
    
    Args:
        prices: DataFrame with price data
        holdings: List of tickers to include in the portfolio
        total_value: Total portfolio value
        
    Returns:
        DataFrame with current positions
    """
    # Get the latest prices
    latest_prices = prices.iloc[-1]
    
    # Create a portfolio DataFrame
    portfolio = pd.DataFrame(index=holdings)
    
    # Assign random weights that sum to 100%
    np.random.seed(42)
    random_weights = np.random.uniform(0.5, 1.5, len(holdings))
    weights = random_weights / random_weights.sum()
    
    # Calculate position values and quantities
    portfolio['weight'] = weights
    portfolio['market_value'] = portfolio['weight'] * total_value
    portfolio['current_price'] = latest_prices[holdings].values
    portfolio['quantity'] = (portfolio['market_value'] / portfolio['current_price']).astype(int)
    
    # Recalculate market value and weight based on actual quantities
    portfolio['market_value'] = portfolio['quantity'] * portfolio['current_price']
    total_value = portfolio['market_value'].sum()
    portfolio['weight'] = portfolio['market_value'] / total_value
    
    # Add lot size for trading
    portfolio['lot'] = 1  # Assume lot size of 1 for simplicity
    
    return portfolio

def plot_portfolio_comparison(current_weights: pd.Series, target_weights: pd.Series, projected_weights: pd.Series):
    """
    Plot a comparison of current, target, and projected portfolio weights.
    
    Args:
        current_weights: Series with current weights
        target_weights: Series with target weights
        projected_weights: Series with projected weights after rebalancing
    """
    # Combine all tickers
    all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index) | set(projected_weights.index)))
    
    # Create a DataFrame with all weights
    comparison = pd.DataFrame(index=all_tickers)
    comparison['Current'] = current_weights
    comparison['Target'] = target_weights
    comparison['Projected'] = projected_weights
    comparison = comparison.fillna(0) * 100  # Convert to percentages
    
    # Sort by target weight
    comparison = comparison.sort_values('Target', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot grouped bars
    x = np.arange(len(comparison))
    width = 0.25
    
    plt.bar(x - width, comparison['Current'], width, label='Current')
    plt.bar(x, comparison['Target'], width, label='Target')
    plt.bar(x + width, comparison['Projected'], width, label='Projected')
    
    plt.xlabel('Ticker')
    plt.ylabel('Weight (%)')
    plt.title('Portfolio Weight Comparison')
    plt.xticks(x, comparison.index, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/portfolio_comparison.png")
    plt.close()

async def main():
    """Main entry point of the script."""
    result = await run_rebalance_demo()
    
    # Create comparison plot
    current_weights = pd.Series(result['current_portfolio']['weight'].to_dict())
    target_weights = result['target_weights']
    projected_weights = pd.Series(result['projected_portfolio']['weight'].to_dict())
    
    plot_portfolio_comparison(current_weights, target_weights, projected_weights)
    logger.info("Portfolio comparison plot saved to results/portfolio_comparison.png")

if __name__ == "__main__":
    asyncio.run(main()) 