from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import date, datetime


@dataclass
class Price:
    """
    Domain entity representing a price point for a security
    """
    ticker: str
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adjusted_close: Optional[float] = None
    volume: Optional[int] = None
    source: str = "unknown"
    timestamp: datetime = None
    
    def __post_init__(self):
        """
        Post-initialization processing.
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
        Create a Price instance from a dictionary.
        
        Args:
            data: Dictionary with price data
            
        Returns:
            Price instance
        """
        return cls(
            ticker=data["ticker"],
            date=data["date"],
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            close=data.get("close"),
            adjusted_close=data.get("adjusted_close") or data.get("adjclose"),
            volume=data.get("volume"),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the price
        """
        return {
            "ticker": self.ticker,
            "date": self.date.strftime("%Y-%m-%d") if isinstance(self.date, date) else self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adjusted_close": self.adjusted_close,
            "volume": self.volume,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        } 