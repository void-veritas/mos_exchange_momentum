from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import date

from .security import Security


@dataclass
class IndexConstituent:
    """
    Domain entity representing a security in an index with its weight
    """
    security: Security
    weight: float
    date: date
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexConstituent':
        """
        Create an IndexConstituent from a dictionary.
        
        Args:
            data: Dictionary with constituent data
            
        Returns:
            IndexConstituent instance
        """
        security_data = {
            "ticker": data["ticker"],
            "name": data.get("name", ""),
            "figi": data.get("figi"),
            "isin": data.get("isin"),
            "currency": data.get("currency", "RUB"),
            "sector": data.get("sector"),
            "exchange": data.get("exchange", "MOEX")
        }
        
        security = Security.from_dict(security_data)
        
        return cls(
            security=security,
            weight=data.get("weight", 0.0),
            date=data["date"] if isinstance(data["date"], date) else 
                  date.fromisoformat(data["date"]) if isinstance(data["date"], str) else
                  date.today()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = self.security.to_dict()
        result.update({
            "weight": self.weight,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date
        })
        return result


@dataclass
class Index:
    """
    Domain entity representing a market index (like MOEX Index)
    """
    index_id: str
    name: str
    date: date
    constituents: List[IndexConstituent] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Index':
        """
        Create an Index from a dictionary.
        
        Args:
            data: Dictionary with index data
            
        Returns:
            Index instance
        """
        constituents = []
        if "constituents" in data:
            for constituent_data in data["constituents"]:
                constituents.append(IndexConstituent.from_dict(constituent_data))
        
        return cls(
            index_id=data["index_id"],
            name=data.get("name", ""),
            date=data["date"] if isinstance(data["date"], date) else 
                 date.fromisoformat(data["date"]) if isinstance(data["date"], str) else
                 date.today(),
            constituents=constituents
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "index_id": self.index_id,
            "name": self.name,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "constituents": [constituent.to_dict() for constituent in self.constituents]
        } 