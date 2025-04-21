from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from datetime import date, datetime
from enum import Enum


class EventType(Enum):
    """
    Types of corporate events that can affect price series
    """
    STOCK_SPLIT = "stock_split"
    DIVIDEND = "dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    NAME_CHANGE = "name_change"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"


@dataclass
class CorporateEvent:
    """
    Domain entity representing a corporate event for a security
    """
    ticker: str
    event_date: date
    event_type: EventType
    event_value: Optional[float] = None
    details: Optional[str] = None
    source: str = "unknown"
    timestamp: datetime = None
    
    def __post_init__(self):
        """
        Post-initialization processing.
        """
        # Convert string date to date object if needed
        if isinstance(self.event_date, str):
            self.event_date = datetime.strptime(self.event_date, "%Y-%m-%d").date()
            
        # Convert string event_type to enum if needed
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)
            
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorporateEvent':
        """
        Create a CorporateEvent instance from a dictionary.
        
        Args:
            data: Dictionary with event data
            
        Returns:
            CorporateEvent instance
        """
        return cls(
            ticker=data["ticker"],
            event_date=data["event_date"],
            event_type=data["event_type"],
            event_value=data.get("event_value"),
            details=data.get("details"),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp")
        ) 