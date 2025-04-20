from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, timedelta

from ..entities.price import Price
from ..entities.security import Security
from ..interfaces.price_repository import PriceRepository
from ..interfaces.data_source import DataSource


class PriceService:
    """
    Domain service for working with price data
    """
    
    def __init__(self, repository: PriceRepository, data_sources: List[DataSource]):
        """
        Initialize price service
        
        Args:
            repository: Repository for storing and retrieving price data
            data_sources: List of data sources to use, in order of preference
        """
        self.repository = repository
        self.data_sources = data_sources
    
    async def get_prices(self, 
                   tickers: List[str],
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None,
                   force_refresh: bool = False) -> Dict[str, List[Price]]:
        """
        Get price data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = date.today()
        elif isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
            
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        
        # Try to get existing data from repository first, unless force_refresh is True
        result = {}
        if not force_refresh:
            existing_data = await self.repository.get_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            # Check if we have complete data for all tickers
            complete = True
            for ticker in tickers:
                if ticker not in existing_data or not existing_data[ticker]:
                    complete = False
                    break
            
            if complete:
                return existing_data
            
            # Store what we already have
            result = existing_data
        
        # Try each data source in order until we get data for all tickers
        missing_tickers = [ticker for ticker in tickers if ticker not in result or not result[ticker]]
        
        if missing_tickers:
            for source in self.data_sources:
                if not missing_tickers:
                    break
                    
                try:
                    new_data = await source.fetch_prices(
                        tickers=missing_tickers,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Store the fetched data
                    all_prices = []
                    for ticker, prices in new_data.items():
                        if prices:
                            # Add to result
                            if ticker not in result:
                                result[ticker] = prices
                            else:
                                # Merge with existing data
                                existing_dates = {p.date.isoformat() if isinstance(p.date, date) else p.date: i 
                                               for i, p in enumerate(result[ticker])}
                                
                                for price in prices:
                                    price_date = price.date.isoformat() if isinstance(price.date, date) else price.date
                                    if price_date not in existing_dates:
                                        result[ticker].append(price)
                            
                            # Add to list for storing in repository
                            all_prices.extend(prices)
                            
                            # Remove ticker from missing list
                            if ticker in missing_tickers:
                                missing_tickers.remove(ticker)
                    
                    # Store in repository
                    if all_prices:
                        await self.repository.store_prices(all_prices)
                        
                except Exception as e:
                    # Log error and continue with next source
                    print(f"Error fetching from {source.name}: {e}")
                    continue
        
        return result
    
    async def get_latest_prices(self, tickers: List[str]) -> Dict[str, Price]:
        """
        Get the latest price for each ticker
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to their latest Price object
        """
        return await self.repository.get_latest_prices(tickers)
    
    async def calculate_returns(self, 
                         ticker: str,
                         start_date: Optional[Union[str, date]] = None,
                         end_date: Optional[Union[str, date]] = None) -> Dict[str, float]:
        """
        Calculate returns for a specific ticker
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with calculated returns
        """
        # Get price data
        prices_data = await self.get_prices(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date
        )
        
        if ticker not in prices_data or not prices_data[ticker]:
            return {
                "daily_return": None,
                "total_return": None,
                "annualized_return": None
            }
        
        # Sort prices by date
        prices = sorted(prices_data[ticker], key=lambda p: p.date)
        
        if len(prices) < 2:
            return {
                "daily_return": None,
                "total_return": None,
                "annualized_return": None
            }
        
        # Calculate returns
        first_price = prices[0].close or prices[0].adjusted_close
        last_price = prices[-1].close or prices[-1].adjusted_close
        prev_price = prices[-2].close or prices[-2].adjusted_close
        
        if not first_price or not last_price or not prev_price:
            return {
                "daily_return": None,
                "total_return": None,
                "annualized_return": None
            }
        
        # Calculate daily return
        daily_return = (last_price / prev_price - 1) * 100
        
        # Calculate total return
        total_return = (last_price / first_price - 1) * 100
        
        # Calculate annualized return
        days = (prices[-1].date - prices[0].date).days
        if days > 0:
            annualized_return = ((last_price / first_price) ** (365 / days) - 1) * 100
        else:
            annualized_return = None
        
        return {
            "daily_return": daily_return,
            "total_return": total_return,
            "annualized_return": annualized_return
        } 