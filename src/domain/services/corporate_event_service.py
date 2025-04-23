from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, timedelta

from ..entities.corporate_event import CorporateEvent, EventType, DividendType
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
    
    async def get_dividends(self,
                       tickers: List[str],
                       dividend_types: Optional[List[DividendType]] = None,
                       start_date: Optional[Union[str, date]] = None,
                       end_date: Optional[Union[str, date]] = None,
                       force_refresh: bool = False) -> Dict[str, List[CorporateEvent]]:
        """
        Get dividend events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            dividend_types: Optional list of dividend types to filter by
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        # First get all dividend events
        all_dividend_events = await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.DIVIDEND,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # If no dividend types filter, return all dividends
        if not dividend_types:
            return all_dividend_events
        
        # Filter by dividend type
        result = {}
        for ticker, events in all_dividend_events.items():
            filtered_events = [
                event for event in events 
                if event.dividend_type in dividend_types
            ]
            if filtered_events:
                result[ticker] = filtered_events
        
        return result
    
    async def get_splits(self,
                    tickers: List[str],
                    start_date: Optional[Union[str, date]] = None,
                    end_date: Optional[Union[str, date]] = None,
                    force_refresh: bool = False) -> Dict[str, List[CorporateEvent]]:
        """
        Get stock split events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        return await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.STOCK_SPLIT,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
    
    async def get_mergers_and_acquisitions(self,
                                     tickers: List[str],
                                     start_date: Optional[Union[str, date]] = None,
                                     end_date: Optional[Union[str, date]] = None,
                                     force_refresh: bool = False) -> Dict[str, List[CorporateEvent]]:
        """
        Get merger and acquisition events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        # Get events for both merger and acquisition types
        merger_events = await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.MERGER,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        acquisition_events = await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.ACQUISITION,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Combine results
        result = {}
        for ticker in set(list(merger_events.keys()) + list(acquisition_events.keys())):
            result[ticker] = []
            if ticker in merger_events:
                result[ticker].extend(merger_events[ticker])
            if ticker in acquisition_events:
                result[ticker].extend(acquisition_events[ticker])
            
            # Sort by date
            if result[ticker]:
                result[ticker].sort(key=lambda e: e.event_date)
        
        return result
    
    async def get_event_by_id(self, event_id: Any) -> Optional[CorporateEvent]:
        """
        Get a specific corporate event by its ID
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            CorporateEvent object if found, None otherwise
        """
        return await self.repository.get_event_by_id(event_id)
    
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
                
                # Handle dividend adjustments if configured
                elif event.event_type == EventType.DIVIDEND and event.event_value and EventType.DIVIDEND in event_types:
                    # Apply dividend adjustment to all prices before the ex-dividend date
                    ex_date = event.ex_dividend_date or event.event_date
                    
                    for price in result[ticker]:
                        if price.date < ex_date:
                            # Adjust based on close price on ex-dividend date
                            if price.close is not None and event.close_price:
                                adjustment_factor = 1 - (event.event_value / event.close_price)
                                
                                if price.open is not None:
                                    price.open *= adjustment_factor
                                if price.high is not None:
                                    price.high *= adjustment_factor
                                if price.low is not None:
                                    price.low *= adjustment_factor
                                if price.close is not None:
                                    price.close *= adjustment_factor
                
        return result 