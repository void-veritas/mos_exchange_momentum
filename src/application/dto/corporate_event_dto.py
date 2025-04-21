from typing import Optional, Dict, Any, List
from datetime import date, datetime
import pandas as pd

from src.domain.entities.corporate_event import CorporateEvent, EventType


class CorporateEventDTO:
    """
    Data Transfer Object for corporate events
    """
    
    def __init__(self, 
                ticker: str,
                event_date: str,
                event_type: str,
                event_value: Optional[float] = None,
                details: Optional[str] = None,
                source: str = "unknown"):
        """
        Initialize corporate event DTO
        
        Args:
            ticker: Ticker symbol
            event_date: Event date (ISO format: YYYY-MM-DD)
            event_type: Event type as string
            event_value: Optional numeric value associated with the event
            details: Optional additional details about the event
            source: Source of the event data
        """
        self.ticker = ticker
        self.event_date = event_date
        self.event_type = event_type
        self.event_value = event_value
        self.details = details
        self.source = source
    
    @classmethod
    def from_entity(cls, entity: CorporateEvent) -> 'CorporateEventDTO':
        """
        Create a CorporateEventDTO from a CorporateEvent entity
        
        Args:
            entity: CorporateEvent domain entity
            
        Returns:
            CorporateEventDTO instance
        """
        return cls(
            ticker=entity.ticker,
            event_date=entity.event_date.isoformat(),
            event_type=entity.event_type.value,
            event_value=entity.event_value,
            details=entity.details,
            source=entity.source
        )
    
    def to_entity(self) -> CorporateEvent:
        """
        Convert the DTO to a domain entity
        
        Returns:
            CorporateEvent domain entity
        """
        return CorporateEvent(
            ticker=self.ticker,
            event_date=self.event_date,
            event_type=self.event_type,
            event_value=self.event_value,
            details=self.details,
            source=self.source
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DTO to a dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "ticker": self.ticker,
            "event_date": self.event_date,
            "event_type": self.event_type,
            "event_value": self.event_value,
            "details": self.details,
            "source": self.source
        }


def events_to_dataframe(events: List[CorporateEvent]) -> pd.DataFrame:
    """
    Convert a list of CorporateEvent objects to a pandas DataFrame
    
    Args:
        events: List of CorporateEvent objects
        
    Returns:
        DataFrame with event data
    """
    if not events:
        return pd.DataFrame()
    
    # Convert to dictionaries
    data = [
        {
            "ticker": e.ticker,
            "event_date": e.event_date.isoformat() if isinstance(e.event_date, date) else e.event_date,
            "event_type": e.event_type.value if isinstance(e.event_type, EventType) else e.event_type,
            "event_value": e.event_value,
            "details": e.details,
            "source": e.source,
        }
        for e in events
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date to datetime
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
    
    return df 