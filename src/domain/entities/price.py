from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import date, datetime


@dataclass
class Price:
    """
    Domain entity representing a price point for a security
    """
    ticker: str
    date: date
    close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    adj_close: Optional[float] = None
    source: str = "unknown"
    timestamp: datetime = None
    
    # Additional metadata fields
    additional_properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Post-initialization processing
        """
        # Convert string date to date object if needed
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d").date()
            
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now()
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Price':
        """
        Create a Price instance from a dictionary
        
        Args:
            data: Dictionary with price data
            
        Returns:
            Price instance
        """
        # Extract known fields
        price_data = {
            "ticker": data["ticker"],
            "date": data["date"],
            "close": data.get("close"),
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "volume": data.get("volume"),
            "adj_close": data.get("adj_close"),
            "source": data.get("source", "unknown"),
            "timestamp": data.get("timestamp")
        }
        
        # Create instance
        instance = cls(**{k: v for k, v in price_data.items() if v is not None})
        
        # Add any additional properties
        for key, value in data.items():
            if key not in price_data and value is not None:
                instance.additional_properties[key] = value
                
        return instance
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the price to a dictionary
        
        Returns:
            Dictionary representation of the price
        """
        # Start with basic fields
        result = {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
        
        # Add price fields if they have values
        if self.close is not None:
            result["close"] = self.close
        if self.open is not None:
            result["open"] = self.open
        if self.high is not None:
            result["high"] = self.high
        if self.low is not None:
            result["low"] = self.low
        if self.volume is not None:
            result["volume"] = self.volume
        if self.adj_close is not None:
            result["adj_close"] = self.adj_close
            
        result["source"] = self.source
        
        # Add any additional properties
        result.update(self.additional_properties)
        
        return result 