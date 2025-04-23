from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from datetime import date

from ..entities.corporate_event import CorporateEvent, EventType, DividendType


class CorporateEventRepository(ABC):
    """
    Interface for accessing corporate event data
    """
    
    @abstractmethod
    async def save_events(self, events: List[CorporateEvent]) -> None:
        """
        Save corporate events to the repository
        
        Args:
            events: List of CorporateEvent objects to save
        """
        pass
    
    @abstractmethod
    async def get_events(self, 
                    tickers: List[str],
                    event_types: Optional[List[EventType]] = None,
                    start_date: Optional[Union[str, date]] = None,
                    end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Get corporate events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        pass
    
    @abstractmethod
    async def get_events_by_type(self,
                           tickers: List[str],
                           event_type: EventType,
                           start_date: Optional[Union[str, date]] = None,
                           end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Get corporate events of a specific type for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_type: Type of events to retrieve
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        pass
    
    @abstractmethod
    async def get_dividends(self,
                      tickers: List[str],
                      dividend_types: Optional[List[DividendType]] = None,
                      start_date: Optional[Union[str, date]] = None,
                      end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Get dividend events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            dividend_types: Optional list of dividend types to filter by
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        pass
    
    @abstractmethod
    async def get_splits(self,
                   tickers: List[str],
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Get stock split events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        pass
    
    @abstractmethod
    async def get_event_by_id(self, event_id: Any) -> Optional[CorporateEvent]:
        """
        Get a specific corporate event by its ID
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            CorporateEvent object if found, None otherwise
        """
        pass
        
    @abstractmethod
    async def delete_events(self,
                       tickers: List[str],
                       event_types: Optional[List[EventType]] = None,
                       start_date: Optional[Union[str, date]] = None,
                       end_date: Optional[Union[str, date]] = None) -> int:
        """
        Delete corporate events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Number of events deleted
        """
        pass 