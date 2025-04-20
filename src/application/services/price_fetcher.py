import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
import pandas as pd

from ...domain.services.price_service import PriceService
from ...domain.entities.price import Price
from ...domain.interfaces.data_source import DataSource
from ...domain.interfaces.price_repository import PriceRepository
from ..dto.price_dto import prices_to_dataframe, PriceDTO
from ..config import Config


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PriceFetcherService:
    """
    Application service for fetching price data
    """
    
    def __init__(self, price_service: PriceService):
        """
        Initialize the price fetcher service
        
        Args:
            price_service: Domain price service
        """
        self.price_service = price_service
    
    async def fetch_prices(self, 
                     tickers: List[str], 
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None,
                     force_refresh: bool = False,
                     as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Fetch prices for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date 
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return prices as pandas DataFrames
            
        Returns:
            Dictionary with price data for each ticker
        """
        # Get domain entities from service
        prices_by_ticker = await self.price_service.get_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrames
        result = {}
        
        for ticker, prices in prices_by_ticker.items():
            if as_dataframe:
                # Convert to DataFrame
                df = prices_to_dataframe(prices)
                result[ticker] = {
                    "ticker": ticker,
                    "price_data": df,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_points": len(df)
                }
            else:
                # Convert to DTOs
                price_dtos = [PriceDTO.from_entity(p) for p in prices]
                result[ticker] = {
                    "ticker": ticker,
                    "prices": price_dtos,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_points": len(price_dtos)
                }
        
        return result
    
    async def get_latest_prices(self, 
                         tickers: List[str],
                         as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get the latest price for each ticker
        
        Args:
            tickers: List of ticker symbols
            as_dataframe: Whether to return prices as a pandas DataFrame
            
        Returns:
            Dictionary with the latest price for each ticker
        """
        # Get domain entities from service
        latest_prices = await self.price_service.get_latest_prices(tickers)
        
        # Convert to DTOs or DataFrame
        result = {}
        
        if as_dataframe:
            # Convert to DataFrame
            prices_list = list(latest_prices.values())
            df = prices_to_dataframe(prices_list)
            return {
                "price_data": df,
                "tickers": tickers,
                "data_points": len(df)
            }
        else:
            # Convert to DTOs
            for ticker, price in latest_prices.items():
                if price:
                    result[ticker] = PriceDTO.from_entity(price)
            
            return result
    
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
        return await self.price_service.calculate_returns(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )