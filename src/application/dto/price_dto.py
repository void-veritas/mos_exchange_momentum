from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import date, datetime
import pandas as pd

from ...domain.entities.price import Price
from ...domain.entities.security import Security


@dataclass
class PriceDTO:
    """Data Transfer Object for price data"""
    ticker: str
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adjusted_close: Optional[float] = None
    volume: Optional[int] = None
    source: str = "unknown"
    
    @classmethod
    def from_entity(cls, price: Price) -> 'PriceDTO':
        """Convert domain entity to DTO"""
        return cls(
            ticker=price.ticker,
            date=price.date.isoformat() if isinstance(price.date, date) else str(price.date),
            open=price.open,
            high=price.high,
            low=price.low,
            close=price.close,
            adjusted_close=price.adjusted_close,
            volume=price.volume,
            source=price.source
        )
    
    def to_entity(self) -> Price:
        """Convert DTO to domain entity"""
        return Price(
            ticker=self.ticker,
            date=self.date,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            adjusted_close=self.adjusted_close,
            volume=self.volume,
            source=self.source
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adjusted_close": self.adjusted_close,
            "volume": self.volume,
            "source": self.source
        }


@dataclass
class SecurityDTO:
    """Data Transfer Object for security data"""
    ticker: str
    name: str
    figi: Optional[str] = None
    isin: Optional[str] = None
    currency: str = "RUB"
    sector: Optional[str] = None
    exchange: str = "MOEX"
    
    @classmethod
    def from_entity(cls, security: Security) -> 'SecurityDTO':
        """Convert domain entity to DTO"""
        return cls(
            ticker=security.ticker,
            name=security.name,
            figi=security.figi,
            isin=security.isin,
            currency=security.currency,
            sector=security.sector,
            exchange=security.exchange
        )
    
    def to_entity(self) -> Security:
        """Convert DTO to domain entity"""
        return Security(
            ticker=self.ticker,
            name=self.name,
            figi=self.figi,
            isin=self.isin,
            currency=self.currency,
            sector=self.sector,
            exchange=self.exchange
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "figi": self.figi,
            "isin": self.isin,
            "currency": self.currency,
            "sector": self.sector,
            "exchange": self.exchange
        }


def prices_to_dataframe(prices: List[Price]) -> pd.DataFrame:
    """
    Convert a list of Price objects to a pandas DataFrame
    
    Args:
        prices: List of Price objects
        
    Returns:
        DataFrame with price data
    """
    if not prices:
        return pd.DataFrame()
    
    # Convert to dictionaries
    data = [
        {
            "ticker": p.ticker,
            "date": p.date.isoformat() if isinstance(p.date, date) else p.date,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "adjusted_close": p.adjusted_close,
            "volume": p.volume,
            "source": p.source,
        }
        for p in prices
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    return df 