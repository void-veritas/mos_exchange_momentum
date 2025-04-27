from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.domain.entities.portfolio import Portfolio
from src.domain.services.portfolio_service import PortfolioService
from src.domain.services.price_service import PriceService
from src.application.config import Config

logger = logging.getLogger(__name__)


class BacktestingService:
    """Service for running backtesting simulations on trading strategies."""
    
    def __init__(
        self,
        portfolio_service: PortfolioService,
        price_service: PriceService,
        config: Config
    ):
        """
        Initialize the backtesting service.
        
        Args:
            portfolio_service: Service for portfolio operations
            price_service: Service for price data operations
            config: Application configuration
        """
        self.portfolio_service = portfolio_service
        self.price_service = price_service
        self.config = config
        self.results = {}
    
    def run_backtest(
        self,
        tickers: List[str],
        factors: pd.DataFrame,
        start_date: str,
        end_date: str,
        rebalance_freq: str = 'M',  # 'D', 'W', 'M', 'Q', 'Y'
        select_by: str = 'industry',
        initial_capital: float = 10000.0,
        benchmark_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a backtest simulation.
        
        Args:
            tickers: List of tickers to include in the universe
            factors: DataFrame with security factors for selection
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            rebalance_freq: Rebalancing frequency ('D'=daily, 'W'=weekly, 'M'=monthly, etc.)
            select_by: Method to select securities
            initial_capital: Initial capital amount
            benchmark_ticker: Optional ticker for benchmark comparison
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date} with {len(tickers)} tickers")
        
        # Convert dates to pandas Timestamp
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Fetch historical prices for all tickers
        all_prices = {}
        for ticker in tqdm(tickers, desc="Fetching price data"):
            try:
                prices = self.price_service.get_prices(ticker, start_date, end_date)
                if not prices.empty:
                    all_prices[ticker] = prices['close']
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        if not all_prices:
            logger.error("No valid price data found for any ticker")
            return {'success': False, 'error': 'No valid price data'}
        
        # Create price DataFrame
        price_df = pd.DataFrame(all_prices)
        price_df = price_df.sort_index()
        
        # Fetch benchmark if provided
        benchmark_prices = None
        if benchmark_ticker:
            try:
                benchmark_data = self.price_service.get_prices(benchmark_ticker, start_date, end_date)
                if not benchmark_data.empty:
                    benchmark_prices = benchmark_data['close']
            except Exception as e:
                logger.warning(f"Error fetching benchmark {benchmark_ticker}: {e}")
        
        # Create portfolio
        portfolio = self.portfolio_service.create_portfolio(factors, select_by)
        
        # Generate rebalance dates
        rebalance_dates = pd.date_range(start=start_ts, end=end_ts, freq=rebalance_freq)
        rebalance_dates = [date for date in rebalance_dates if date in price_df.index]
        
        if not rebalance_dates:
            logger.error("No valid rebalance dates in the provided date range")
            return {'success': False, 'error': 'No valid rebalance dates'}
        
        # Initialize results storage
        portfolio_values = pd.Series(index=price_df.index, dtype=float)
        portfolio_values.iloc[0] = initial_capital
        holdings = {}
        weights_history = []
        
        # Run simulation
        current_holdings = {}
        last_rebalance_idx = 0
        
        for i, current_date in enumerate(price_df.index):
            if i == 0:
                continue  # Skip the first day
                
            prev_date = price_df.index[i-1]
            
            # Check if we need to rebalance
            should_rebalance = any(rd == current_date for rd in rebalance_dates)
            
            if should_rebalance:
                logger.debug(f"Rebalancing portfolio on {current_date}")
                
                # Select assets based on portfolio strategy
                selected_assets = portfolio.select(price_df.loc[:current_date])
                
                # Skip if no assets selected
                if not selected_assets:
                    logger.warning(f"No assets selected on {current_date}")
                    portfolio_values.loc[current_date] = portfolio_values.loc[prev_date]
                    continue
                
                # Allocate weights
                weights = portfolio.allocate(price_df.loc[:current_date], selected_assets)
                
                # Record weights
                weights_history.append({
                    'date': current_date,
                    'weights': weights.to_dict()
                })
                
                # Calculate cash needed for each position
                current_capital = portfolio_values.loc[prev_date]
                new_holdings = {}
                
                for asset, weight in weights.items():
                    if asset in price_df.columns and not pd.isna(price_df.loc[current_date, asset]):
                        price = price_df.loc[current_date, asset]
                        position_value = current_capital * weight
                        shares = position_value / price
                        new_holdings[asset] = shares
                
                current_holdings = new_holdings
                holdings[str(current_date.date())] = new_holdings
                last_rebalance_idx = i
            
            # Calculate portfolio value
            portfolio_value = 0.0
            for asset, shares in current_holdings.items():
                if asset in price_df.columns and current_date in price_df.index:
                    if not pd.isna(price_df.loc[current_date, asset]):
                        price = price_df.loc[current_date, asset]
                        position_value = shares * price
                        portfolio_value += position_value
            
            portfolio_values.loc[current_date] = portfolio_value
        
        # Calculate returns
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_prices is not None:
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Align benchmark returns with portfolio returns
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_idx]
            benchmark_returns = benchmark_returns.loc[common_idx]
        
        # Calculate performance metrics
        metrics = self.portfolio_service.calculate_metrics(portfolio_returns)
        
        # Calculate benchmark metrics if available
        benchmark_metrics = None
        if benchmark_returns is not None:
            benchmark_metrics = self.portfolio_service.calculate_metrics(benchmark_returns)
        
        # Store results
        results = {
            'success': True,
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'holdings': holdings,
            'weights_history': weights_history,
            'metrics': metrics,
            'benchmark_returns': benchmark_returns,
            'benchmark_metrics': benchmark_metrics,
        }
        
        self.results = results
        return results
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results or not self.results.get('success', False):
            logger.error("No valid backtest results to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Get data
        portfolio_values = self.results['portfolio_values']
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results.get('benchmark_returns')
        
        # Plot portfolio value
        portfolio_values.plot(ax=axes[0], color='blue', lw=2)
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)
        
        # Plot cumulative returns
        cum_returns = (1 + portfolio_returns).cumprod() - 1
        cum_returns.plot(ax=axes[1], color='green', lw=2, label='Strategy')
        
        if benchmark_returns is not None:
            cum_bench_returns = (1 + benchmark_returns).cumprod() - 1
            cum_bench_returns.plot(ax=axes[1], color='red', lw=2, label='Benchmark')
            
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot drawdowns
        def calculate_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            return drawdown
        
        drawdown = calculate_drawdown(portfolio_returns)
        drawdown.plot(ax=axes[2], color='darkred', lw=2, label='Strategy')
        
        if benchmark_returns is not None:
            bench_drawdown = calculate_drawdown(benchmark_returns)
            bench_drawdown.plot(ax=axes[2], color='orange', lw=2, label='Benchmark')
            
        axes[2].set_title('Drawdowns')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def export_results(self, output_path: str) -> None:
        """
        Export backtest results to Excel.
        
        Args:
            output_path: Path to save the Excel file
        """
        if not self.results or not self.results.get('success', False):
            logger.error("No valid backtest results to export")
            return
        
        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            # Portfolio values
            self.results['portfolio_values'].to_excel(writer, sheet_name='Portfolio Values')
            
            # Portfolio returns
            self.results['portfolio_returns'].to_excel(writer, sheet_name='Portfolio Returns')
            
            # Benchmark returns if available
            if self.results.get('benchmark_returns') is not None:
                self.results['benchmark_returns'].to_excel(writer, sheet_name='Benchmark Returns')
            
            # Metrics
            pd.Series(self.results['metrics']).to_excel(writer, sheet_name='Metrics')
            
            # Benchmark metrics if available
            if self.results.get('benchmark_metrics') is not None:
                pd.Series(self.results['benchmark_metrics']).to_excel(
                    writer, sheet_name='Benchmark Metrics'
                )
            
            # Weights history
            weights_df = pd.DataFrame([
                {**{'date': item['date']}, **item['weights']}
                for item in self.results['weights_history']
            ])
            if not weights_df.empty:
                weights_df.set_index('date', inplace=True)
                weights_df.to_excel(writer, sheet_name='Weights History')
            
            # Summary
            summary = pd.Series({
                'Start Date': self.results['portfolio_values'].index[0],
                'End Date': self.results['portfolio_values'].index[-1],
                'Initial Value': self.results['portfolio_values'].iloc[0],
                'Final Value': self.results['portfolio_values'].iloc[-1],
                'Total Return': f"{self.results['metrics']['total_return']:.2%}",
                'Annualized Return': f"{self.results['metrics']['annualized_return']:.2%}",
                'Volatility': f"{self.results['metrics']['volatility']:.2%}",
                'Sharpe Ratio': f"{self.results['metrics']['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{self.results['metrics']['max_drawdown']:.2%}",
            })
            summary.to_excel(writer, sheet_name='Summary')
        
        logger.info(f"Exported backtest results to {output_path}") 