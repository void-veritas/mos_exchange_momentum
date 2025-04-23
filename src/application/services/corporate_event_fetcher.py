from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime

from src.domain.services.corporate_event_service import CorporateEventService
from src.domain.entities.corporate_event import EventType, DividendType, CorporateEvent
from src.application.dto.corporate_event_dto import CorporateEventDTO, events_to_dataframe


class CorporateEventFetcherService:
    """
    Application service for fetching corporate event data
    """
    
    def __init__(self, corporate_event_service: CorporateEventService):
        """
        Initialize corporate event fetcher service
        
        Args:
            corporate_event_service: Domain service for corporate events
        """
        self.corporate_event_service = corporate_event_service
    
    async def get_events(self, 
                   tickers: List[str],
                   event_types: Optional[List[str]] = None,
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None,
                   force_refresh: bool = False,
                   as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get corporate event data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by (as strings)
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return events as a pandas DataFrame
            
        Returns:
            Dictionary with corporate event data
        """
        # Convert string event types to enum
        domain_event_types = None
        if event_types:
            try:
                domain_event_types = [EventType(et) for et in event_types]
            except ValueError as e:
                raise ValueError(f"Invalid event type: {e}")
        
        # Get domain entities from service
        events_data = await self.corporate_event_service.get_events(
            tickers=tickers,
            event_types=domain_event_types,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrame
        result = {}
        
        if as_dataframe:
            # Flatten into a single list of events
            all_events = []
            for ticker, events in events_data.items():
                all_events.extend(events)
                
            # Convert to DataFrame
            df = events_to_dataframe(all_events)
            return {
                "event_data": df,
                "tickers": tickers,
                "data_points": len(df)
            }
        else:
            # Convert to DTOs
            for ticker, events in events_data.items():
                if events:
                    result[ticker] = [CorporateEventDTO.from_entity(event) for event in events]
                else:
                    result[ticker] = []
            
            return result
    
    async def get_events_by_type(self,
                           tickers: List[str],
                           event_type: str,
                           start_date: Optional[Union[str, date]] = None,
                           end_date: Optional[Union[str, date]] = None,
                           force_refresh: bool = False,
                           as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get corporate events of a specific type for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            event_type: Type of events to retrieve (as string)
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return events as a pandas DataFrame
            
        Returns:
            Dictionary with corporate event data filtered by type
        """
        # Convert string event type to enum
        try:
            domain_event_type = EventType(event_type)
        except ValueError:
            raise ValueError(f"Invalid event type: {event_type}")
        
        # Get events by type
        events_data = await self.corporate_event_service.get_events_by_type(
            tickers=tickers,
            event_type=domain_event_type,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrame
        result = {}
        
        if as_dataframe:
            # Flatten into a single list of events
            all_events = []
            for ticker, events in events_data.items():
                all_events.extend(events)
                
            # Convert to DataFrame
            df = events_to_dataframe(all_events)
            return {
                "event_data": df,
                "tickers": tickers,
                "event_type": event_type,
                "data_points": len(df)
            }
        else:
            # Convert to DTOs
            for ticker, events in events_data.items():
                if events:
                    result[ticker] = [CorporateEventDTO.from_entity(event) for event in events]
                else:
                    result[ticker] = []
            
            return result
    
    async def get_dividends(self,
                       tickers: List[str],
                       dividend_types: Optional[List[str]] = None,
                       start_date: Optional[Union[str, date]] = None,
                       end_date: Optional[Union[str, date]] = None,
                       force_refresh: bool = False,
                       as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get dividend events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            dividend_types: Optional list of dividend types to filter by (as strings)
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return events as a pandas DataFrame
            
        Returns:
            Dictionary with dividend data
        """
        # Convert string dividend types to enum if provided
        domain_dividend_types = None
        if dividend_types:
            try:
                domain_dividend_types = [DividendType(dt) for dt in dividend_types]
            except ValueError as e:
                raise ValueError(f"Invalid dividend type: {e}")
        
        # Get dividend events
        events_data = await self.corporate_event_service.get_dividends(
            tickers=tickers,
            dividend_types=domain_dividend_types,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrame
        result = {}
        
        if as_dataframe:
            # Flatten into a single list of events
            all_events = []
            for ticker, events in events_data.items():
                all_events.extend(events)
                
            # Convert to DataFrame
            df = events_to_dataframe(all_events)
            return {
                "event_data": df,
                "tickers": tickers,
                "event_type": "dividend",
                "dividend_types": dividend_types,
                "data_points": len(df)
            }
        else:
            # Convert to DTOs
            for ticker, events in events_data.items():
                if events:
                    result[ticker] = [CorporateEventDTO.from_entity(event) for event in events]
                else:
                    result[ticker] = []
            
            return result
    
    async def get_splits(self,
                    tickers: List[str],
                    start_date: Optional[Union[str, date]] = None,
                    end_date: Optional[Union[str, date]] = None,
                    force_refresh: bool = False,
                    as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get stock split events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return events as a pandas DataFrame
            
        Returns:
            Dictionary with stock split data
        """
        # Get split events
        events_data = await self.corporate_event_service.get_splits(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrame
        return self._convert_events_result(
            events_data=events_data,
            tickers=tickers,
            event_type="stock_split",
            as_dataframe=as_dataframe
        )
    
    async def get_mergers_and_acquisitions(self,
                                     tickers: List[str],
                                     start_date: Optional[Union[str, date]] = None,
                                     end_date: Optional[Union[str, date]] = None,
                                     force_refresh: bool = False,
                                     as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Get merger and acquisition events for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            force_refresh: Whether to force fetching from data sources
            as_dataframe: Whether to return events as a pandas DataFrame
            
        Returns:
            Dictionary with merger and acquisition data
        """
        # Get M&A events
        events_data = await self.corporate_event_service.get_mergers_and_acquisitions(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Convert to DTOs or DataFrame
        return self._convert_events_result(
            events_data=events_data,
            tickers=tickers,
            event_type="mergers_acquisitions",
            as_dataframe=as_dataframe
        )
    
    async def get_event_by_id(self, event_id: Any) -> Optional[CorporateEventDTO]:
        """
        Get a specific corporate event by its ID
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            CorporateEventDTO object if found, None otherwise
        """
        event = await self.corporate_event_service.get_event_by_id(event_id)
        if event:
            return CorporateEventDTO.from_entity(event)
        return None
            
    async def apply_corporate_events_to_prices(self,
                                        price_data: Dict[str, Any],
                                        event_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply corporate events to price data
        
        Args:
            price_data: Dictionary with price data (as returned by PriceFetcherService)
            event_types: Optional list of event types to apply (as strings)
            
        Returns:
            Dictionary with adjusted price data
        """
        # Convert string event types to enum if provided
        domain_event_types = None
        if event_types:
            try:
                domain_event_types = [EventType(et) for et in event_types]
            except ValueError as e:
                raise ValueError(f"Invalid event type: {e}")
        
        # Apply events to prices
        adjusted_data = await self.corporate_event_service.apply_corporate_events_to_prices(
            prices=price_data,
            event_types=domain_event_types
        )
        
        # Return the adjusted data with the same format as the input
        return adjusted_data
    
    def _convert_events_result(self,
                               events_data: Dict[str, List[CorporateEvent]],
                               tickers: List[str],
                               event_type: str,
                               as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Helper method to convert event results to DTOs or DataFrame
        
        Args:
            events_data: Dictionary with events data from domain service
            tickers: List of ticker symbols
            event_type: Type of events (for metadata)
            as_dataframe: Whether to return as DataFrame
            
        Returns:
            Converted result
        """
        if as_dataframe:
            # Flatten into a single list of events
            all_events = []
            for ticker, events in events_data.items():
                all_events.extend(events)
                
            # Convert to DataFrame
            df = events_to_dataframe(all_events)
            return {
                "event_data": df,
                "tickers": tickers,
                "event_type": event_type,
                "data_points": len(df)
            }
        else:
            # Convert to DTOs
            result = {}
            for ticker, events in events_data.items():
                if events:
                    result[ticker] = [CorporateEventDTO.from_entity(event) for event in events]
                else:
                    result[ticker] = []
            
            return result 