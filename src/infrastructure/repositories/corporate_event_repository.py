import logging
import json
import os
from typing import List, Dict, Optional, Union, Any
from datetime import date, datetime

from ...domain.interfaces.corporate_event_repository import CorporateEventRepository
from ...domain.entities.corporate_event import CorporateEvent, EventType, DividendType


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InMemoryCorporateEventRepository(CorporateEventRepository):
    """
    In-memory implementation of CorporateEventRepository
    Uses a JSON file for persistence between runs
    """
    
    def __init__(self, storage_file: Optional[str] = None):
        """
        Initialize repository
        
        Args:
            storage_file: Path to JSON file for persistence (optional)
        """
        self.storage_file = storage_file or "data/corporate_events.json"
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self._load_events()
    
    async def save_events(self, events: List[CorporateEvent]) -> None:
        """
        Save corporate events to the repository
        
        Args:
            events: List of CorporateEvent objects to save
        """
        for event in events:
            ticker = event.ticker
            
            # Convert to dictionary for storage
            event_dict = event.to_dict()
            
            # Add to in-memory storage
            if ticker not in self.events:
                self.events[ticker] = []
            
            # Check if this event already exists
            existing = [
                e for e in self.events[ticker] 
                if (e.get("event_date") == event_dict["event_date"] and
                    e.get("event_type") == event_dict["event_type"] and
                    e.get("event_value") == event_dict["event_value"])
            ]
            
            if not existing:
                self.events[ticker].append(event_dict)
        
        # Persist to file
        self._save_events()
        logger.info(f"Saved {len(events)} events to repository")
    
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
        # Normalize dates
        if start_date and isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        if end_date and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Filter events
        result = {}
        for ticker in tickers:
            if ticker not in self.events:
                result[ticker] = []
                continue
            
            ticker_events = []
            for event_dict in self.events[ticker]:
                # Convert dates for comparison
                event_date_str = event_dict.get("event_date")
                if event_date_str:
                    if isinstance(event_date_str, str):
                        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
                    else:
                        event_date = event_date_str
                else:
                    continue
                
                # Check date range
                if start_date and event_date < start_date:
                    continue
                
                if end_date and event_date > end_date:
                    continue
                
                # Check event type
                if event_types:
                    event_type_str = event_dict.get("event_type")
                    try:
                        event_type = EventType(event_type_str)
                        if event_type not in event_types:
                            continue
                    except (ValueError, TypeError):
                        continue
                
                # Create event object
                event = CorporateEvent.from_dict(event_dict)
                ticker_events.append(event)
            
            result[ticker] = ticker_events
        
        return result
    
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
        return await self.get_events(
            tickers=tickers,
            event_types=[event_type],
            start_date=start_date,
            end_date=end_date
        )
    
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
        # First get all dividend events
        dividend_events = await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.DIVIDEND,
            start_date=start_date,
            end_date=end_date
        )
        
        # If no dividend types filter, return all dividends
        if not dividend_types:
            return dividend_events
        
        # Filter by dividend type
        result = {}
        for ticker, events in dividend_events.items():
            filtered_events = [
                event for event in events 
                if event.dividend_type in dividend_types
            ]
            result[ticker] = filtered_events
        
        return result
    
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
        return await self.get_events_by_type(
            tickers=tickers,
            event_type=EventType.STOCK_SPLIT,
            start_date=start_date,
            end_date=end_date
        )
    
    async def get_event_by_id(self, event_id: Any) -> Optional[CorporateEvent]:
        """
        Get a specific corporate event by its ID
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            CorporateEvent object if found, None otherwise
        """
        # This implementation doesn't use IDs, so we return None
        return None
    
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
        # Normalize dates
        if start_date and isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        if end_date and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Delete events
        deleted_count = 0
        for ticker in tickers:
            if ticker not in self.events:
                continue
            
            # Filter events to keep
            events_to_keep = []
            for event_dict in self.events[ticker]:
                should_delete = False
                
                # Convert dates for comparison
                event_date_str = event_dict.get("event_date")
                if event_date_str:
                    if isinstance(event_date_str, str):
                        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
                    else:
                        event_date = event_date_str
                else:
                    events_to_keep.append(event_dict)
                    continue
                
                # Check date range
                if start_date and event_date < start_date:
                    events_to_keep.append(event_dict)
                    continue
                
                if end_date and event_date > end_date:
                    events_to_keep.append(event_dict)
                    continue
                
                # Check event type
                if event_types:
                    event_type_str = event_dict.get("event_type")
                    try:
                        event_type = EventType(event_type_str)
                        if event_type not in event_types:
                            events_to_keep.append(event_dict)
                            continue
                    except (ValueError, TypeError):
                        events_to_keep.append(event_dict)
                        continue
                
                # If we got here, the event should be deleted
                deleted_count += 1
            
            if events_to_keep:
                self.events[ticker] = events_to_keep
            else:
                del self.events[ticker]
        
        # Persist changes
        self._save_events()
        
        return deleted_count
    
    def _load_events(self) -> None:
        """
        Load events from storage file
        """
        if not os.path.exists(self.storage_file):
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            return
        
        try:
            with open(self.storage_file, 'r') as f:
                self.events = json.load(f)
            logger.info(f"Loaded events from {self.storage_file}")
        except Exception as e:
            logger.error(f"Error loading events from {self.storage_file}: {e}")
    
    def _save_events(self) -> None:
        """
        Save events to storage file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            with open(self.storage_file, 'w') as f:
                json.dump(self.events, f, indent=2)
            logger.info(f"Saved events to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving events to {self.storage_file}: {e}")


# Default implementation
CorporateEventRepository = InMemoryCorporateEventRepository 