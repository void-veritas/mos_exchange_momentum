from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import date


@dataclass
class Security:
    """
    Domain entity representing a security (stock, bond, etc.)
    """
    ticker: str
    name: str
    figi: Optional[str] = None
    isin: Optional[str] = None
    currency: str = "RUB"
    sector: Optional[str] = None
    exchange: str = "MOEX"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Security':
        """
        Create a Security instance from a dictionary.
        
        Args:
            data: Dictionary with security data
            
        Returns:
            Security instance
        """
        return cls(
            ticker=data["ticker"],
            name=data.get("name", ""),
            figi=data.get("figi"),
            isin=data.get("isin"),
            currency=data.get("currency", "RUB"),
            sector=data.get("sector"),
            exchange=data.get("exchange", "MOEX")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the security
        """
        return {
            "ticker": self.ticker,
            "name": self.name,
            "figi": self.figi,
            "isin": self.isin,
            "currency": self.currency,
            "sector": self.sector,
            "exchange": self.exchange
        } 