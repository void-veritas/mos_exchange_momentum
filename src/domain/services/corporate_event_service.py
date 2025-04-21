from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, timedelta

from ..entities.corporate_event import CorporateEvent, EventType
from ..interfaces.corporate_event_repository import CorporateEventRepository
from ..interfaces.data_source import DataSource


class CorporateEventService:
    """
    Domain service for working with corporate event data
    """
    
    def __init__(self, repository: CorporateEventRepository, data_sources: List[DataSource]):
        """
        Initialize corporate event service
        
        Args:
            repository: Repository for storing and retrieving corporate event data
            data_sources: List of data sources to use, in order of preference
        """
        self.repository = repository
        self.data_sources = data_sources
    
    async def get_events(self, 
                   tickers: List[str],
                   event_types: Optional[List[EventType]] = None,
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None,
                   force_refresh: bool = False) -> Dict[str, List[CorporateEvent]]:
        """
        Get corporate event data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = date.today()
        elif isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
            
        if start_date is None:
            # Default to 1 year ago
            start_date = end_date - timedelta(days=365)
        elif isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        
        # Try to get existing data from repository first, unless force_refresh is True
        result = {}
        if not force_refresh:
            existing_data = await self.repository.get_events(
                tickers=tickers,
                event_types=event_types,
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
        
        for source in self.data_sources:
            if not missing_tickers:
                break
                
            try:
                # Fetch from data source
                fetched_data = await source.fetch_corporate_events(
                    tickers=missing_tickers,
                    event_types=event_types,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Store in result
                for ticker, events in fetched_data.items():
                    if events:
                        # Save to repository
                        await self.repository.save_events(events)
                        
                        # Add to result
                        if ticker not in result:
                            result[ticker] = []
                        result[ticker].extend(events)
                
                # Update missing tickers
                missing_tickers = [ticker for ticker in missing_tickers 
                                  if ticker not in fetched_data or not fetched_data[ticker]]
                
            except Exception as e:
                # Log error and continue with next source
                print(f"Error fetching from {source.name}: {e}")
                continue
        
        return result
    
    async def get_events_by_type(self,
                           tickers: List[str],
                           event_type: EventType,
                           start_date: Optional[Union[str, date]] = None,
                           end_date: Optional[Union[str, date]] = None,
                           force_refresh: bool = False) -> Dict[str, List[CorporateEvent]]:
        """
        Get corporate events of a specific type for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_type: Type of events to retrieve
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects filtered by type
        """
        # Get all events
        all_events = await self.get_events(
            tickers=tickers,
            event_types=[event_type],
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        return all_events
    
    async def apply_corporate_events_to_prices(self,
                                        prices: Dict[str, List[Any]],
                                        event_types: Optional[List[EventType]] = None) -> Dict[str, List[Any]]:
        """
        Apply corporate events to price data
        
        Args:
            prices: Dictionary mapping ticker symbols to lists of Price objects
            event_types: Optional list of event types to apply
            
        Returns:
            Dictionary with adjusted price data
        """
        # If no event types specified, apply splits and dividends by default
        if event_types is None:
            event_types = [EventType.STOCK_SPLIT, EventType.DIVIDEND]
        
        # Get the tickers and date range from the price data
        tickers = list(prices.keys())
        if not tickers:
            return prices
        
        # Find min and max dates
        min_date = None
        max_date = None
        
        for ticker, price_list in prices.items():
            for price in price_list:
                if min_date is None or price.date < min_date:
                    min_date = price.date
                if max_date is None or price.date > max_date:
                    max_date = price.date
        
        if min_date is None or max_date is None:
            return prices
            
        # Get relevant corporate events
        events = await self.get_events(
            tickers=tickers,
            event_types=event_types,
            start_date=min_date,
            end_date=max_date
        )
        
        # Apply events to prices
        result = {ticker: list(price_list) for ticker, price_list in prices.items()}
        
        for ticker, event_list in events.items():
            if ticker not in result:
                continue
                
            # Sort events by date (newest first)
            sorted_events = sorted(event_list, key=lambda e: e.event_date, reverse=True)
            
            for event in sorted_events:
                # Apply event based on type
                if event.event_type == EventType.STOCK_SPLIT and event.event_value:
                    # Apply split to all prices before the event date
                    for price in result[ticker]:
                        if price.date < event.event_date:
                            if price.open is not None:
                                price.open /= event.event_value
                            if price.high is not None:
                                price.high /= event.event_value
                            if price.low is not None:
                                price.low /= event.event_value
                            if price.close is not None:
                                price.close /= event.event_value
                            if price.volume is not None:
                                price.volume *= event.event_value
                
                # Handle other event types as needed
                
        return result 