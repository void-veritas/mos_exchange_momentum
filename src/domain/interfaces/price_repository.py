from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime

from ..entities.price import Price
from ..entities.security import Security


class PriceRepository(ABC):
    """
    Interface for price data repositories
    """
    
    @abstractmethod
    async def store_prices(self, prices: List[Price]) -> None:
        """
        Store a list of price entries
        
        Args:
            prices: List of Price objects to store
        """
        pass
    
    @abstractmethod
    async def get_prices(self, 
                   tickers: List[str], 
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None,
                   source: Optional[str] = None) -> Dict[str, List[Price]]:
        """
        Retrieve prices for the specified tickers and date range
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        pass
    
    @abstractmethod
    async def get_latest_prices(self, 
                         tickers: List[str],
                         source: Optional[str] = None) -> Dict[str, Price]:
        """
        Get the latest price for each ticker
        
        Args:
            tickers: List of ticker symbols
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to their latest Price object
        """
        pass
    
    @abstractmethod
    async def delete_prices(self,
                     tickers: List[str],
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None,
                     source: Optional[str] = None) -> int:
        """
        Delete price entries matching the criteria
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            source: Optional source to filter by
            
        Returns:
            Number of deleted entries
        """
        pass


class SecurityRepository(ABC):
    """
    Interface for security metadata repositories
    """
    
    @abstractmethod
    async def store_securities(self, securities: List[Security]) -> None:
        """
        Store a list of securities
        
        Args:
            securities: List of Security objects to store
        """
        pass
    
    @abstractmethod
    async def get_securities(self, 
                      tickers: Optional[List[str]] = None) -> List[Security]:
        """
        Retrieve securities matching the criteria
        
        Args:
            tickers: Optional list of ticker symbols to filter by
            
        Returns:
            List of Security objects
        """
        pass
    
    @abstractmethod
    async def get_security(self, ticker: str) -> Optional[Security]:
        """
        Get a single security by ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        pass 