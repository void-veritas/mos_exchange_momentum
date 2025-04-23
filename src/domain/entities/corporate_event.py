from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
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
    RIGHTS_ISSUE = "rights_issue"
    BUYBACK = "buyback"
    IPO = "ipo"
    BANKRUPTCY = "bankruptcy"


class DividendType(Enum):
    """
    Types of dividend payments
    """
    REGULAR = "regular"
    SPECIAL = "special"
    INTERIM = "interim"
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"


@dataclass
class CorporateEvent:
    """
    Domain entity representing a corporate event for a security
    """
    ticker: str
    event_date: date  # Main event date
    event_type: EventType
    event_value: Optional[float] = None  # Primary value (dividend amount, split ratio)
    details: Optional[str] = None
    source: str = "unknown"
    timestamp: datetime = None
    
    # Additional metadata fields for all events
    figi: Optional[str] = None  # Financial Instrument Global Identifier
    isin: Optional[str] = None  # International Securities Identification Number
    
    # Specific fields for dividends
    dividend_type: Optional[DividendType] = None
    payment_date: Optional[date] = None
    declared_date: Optional[date] = None
    ex_dividend_date: Optional[date] = None  # Last buy date
    record_date: Optional[date] = None
    currency: Optional[str] = None
    yield_value: Optional[float] = None
    close_price: Optional[float] = None
    regularity: Optional[str] = None
    
    # Fields for splits and other corporate actions
    effective_date: Optional[date] = None
    announcement_date: Optional[date] = None
    
    # Additional properties as a catch-all for event-specific data
    additional_properties: Dict[str, Any] = field(default_factory=dict)
    
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
            
        # Handle dividend_type if provided
        if self.dividend_type is not None and isinstance(self.dividend_type, str):
            try:
                self.dividend_type = DividendType(self.dividend_type)
            except ValueError:
                # If not a valid enum value, set to None
                self.dividend_type = None
                
        # Convert other dates if provided as strings
        date_fields = [
            "payment_date", "declared_date", "ex_dividend_date",
            "record_date", "effective_date", "announcement_date"
        ]
        
        for field_name in date_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, str):
                try:
                    setattr(self, field_name, datetime.strptime(field_value, "%Y-%m-%d").date())
                except (ValueError, TypeError):
                    # If invalid date format, keep as is
                    pass
            
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
        # Extract known fields
        event_data = {
            "ticker": data["ticker"],
            "event_date": data["event_date"],
            "event_type": data["event_type"],
            "event_value": data.get("event_value"),
            "details": data.get("details"),
            "source": data.get("source", "unknown"),
            "timestamp": data.get("timestamp"),
            "figi": data.get("figi"),
            "isin": data.get("isin"),
            "dividend_type": data.get("dividend_type"),
            "payment_date": data.get("payment_date"),
            "declared_date": data.get("declared_date"),
            "ex_dividend_date": data.get("ex_dividend_date") or data.get("last_buy_date"),
            "record_date": data.get("record_date"),
            "currency": data.get("currency"),
            "yield_value": data.get("yield_value"),
            "close_price": data.get("close_price"),
            "regularity": data.get("regularity"),
            "effective_date": data.get("effective_date"),
            "announcement_date": data.get("announcement_date")
        }
        
        # Create instance
        instance = cls(**{k: v for k, v in event_data.items() if v is not None})
        
        # Add any additional properties
        for key, value in data.items():
            if key not in event_data and value is not None:
                instance.additional_properties[key] = value
                
        return instance
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        # Start with basic fields
        result = {
            "ticker": self.ticker,
            "event_date": self.event_date.isoformat() if isinstance(self.event_date, date) else self.event_date,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "event_value": self.event_value,
            "details": self.details,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
        
        # Add specialized fields if they have values
        if self.figi:
            result["figi"] = self.figi
        if self.isin:
            result["isin"] = self.isin
            
        # Add dividend-specific fields
        if self.event_type == EventType.DIVIDEND:
            if self.dividend_type:
                result["dividend_type"] = self.dividend_type.value if isinstance(self.dividend_type, DividendType) else self.dividend_type
            if self.payment_date:
                result["payment_date"] = self.payment_date.isoformat() if isinstance(self.payment_date, date) else self.payment_date
            if self.declared_date:
                result["declared_date"] = self.declared_date.isoformat() if isinstance(self.declared_date, date) else self.declared_date
            if self.ex_dividend_date:
                result["ex_dividend_date"] = self.ex_dividend_date.isoformat() if isinstance(self.ex_dividend_date, date) else self.ex_dividend_date
            if self.record_date:
                result["record_date"] = self.record_date.isoformat() if isinstance(self.record_date, date) else self.record_date
            if self.currency:
                result["currency"] = self.currency
            if self.yield_value is not None:
                result["yield_value"] = self.yield_value
            if self.close_price is not None:
                result["close_price"] = self.close_price
            if self.regularity:
                result["regularity"] = self.regularity
                
        # Add other specialized fields
        date_fields = ["effective_date", "announcement_date"]
        for field_name in date_fields:
            field_value = getattr(self, field_name)
            if field_value is not None:
                result[field_name] = field_value.isoformat() if isinstance(field_value, date) else field_value
                
        # Add any additional properties
        result.update(self.additional_properties)
        
        return result 