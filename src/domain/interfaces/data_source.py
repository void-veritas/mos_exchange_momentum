from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime

from ..entities.price import Price
from ..entities.security import Security
from ..entities.index import Index
from ..entities.corporate_event import CorporateEvent, EventType


class DataSource(ABC):
    """
    Interface for external data sources
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the data source
        
        Returns:
            Data source name
        """
        pass
    
    @abstractmethod
    async def fetch_prices(self, 
                     tickers: List[str],
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None) -> Dict[str, List[Price]]:
        """
        Fetch price data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        pass
    
    @abstractmethod
    async def fetch_security_info(self, ticker: str) -> Optional[Security]:
        """
        Fetch metadata for a specific security
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        pass
    
    async def fetch_corporate_events(self,
                               tickers: List[str],
                               event_types: Optional[List[EventType]] = None,
                               start_date: Optional[Union[str, date]] = None,
                               end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Fetch corporate events data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        # Default implementation returns empty dict
        # Subclasses should override if they support corporate events
        return {ticker: [] for ticker in tickers}


class IndexDataSource(ABC):
    """
    Interface for index data sources
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the data source
        
        Returns:
            Data source name
        """
        pass
    
    @abstractmethod
    async def fetch_index_composition(self,
                               index_id: str,
                               date_str: Optional[str] = None) -> Optional[Index]:
        """
        Fetch index composition for a specific date
        
        Args:
            index_id: Index identifier
            date_str: Date string in ISO format (YYYY-MM-DD)
            
        Returns:
            Index object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def fetch_index_timeseries(self,
                              index_id: str,
                              start_date: Optional[Union[str, date]] = None,
                              end_date: Optional[Union[str, date]] = None,
                              frequency: str = "monthly") -> Dict[str, Index]:
        """
        Fetch index composition over time
        
        Args:
            index_id: Index identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency ("daily", "weekly", "monthly")
            
        Returns:
            Dictionary mapping dates (as strings) to Index objects
        """
        pass 