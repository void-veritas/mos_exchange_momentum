from typing import Optional, Dict, Any, List
from datetime import date, datetime
import pandas as pd

from src.domain.entities.corporate_event import CorporateEvent, EventType, DividendType


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
                source: str = "unknown",
                dividend_type: Optional[str] = None,
                payment_date: Optional[str] = None,
                declared_date: Optional[str] = None,
                ex_dividend_date: Optional[str] = None,
                record_date: Optional[str] = None,
                currency: Optional[str] = None,
                yield_value: Optional[float] = None,
                close_price: Optional[float] = None,
                figi: Optional[str] = None,
                isin: Optional[str] = None,
                regularity: Optional[str] = None,
                effective_date: Optional[str] = None,
                announcement_date: Optional[str] = None,
                additional_properties: Optional[Dict[str, Any]] = None):
        """
        Initialize corporate event DTO
        
        Args:
            ticker: Ticker symbol
            event_date: Event date (ISO format: YYYY-MM-DD)
            event_type: Event type as string
            event_value: Optional numeric value associated with the event
            details: Optional additional details about the event
            source: Source of the event data
            dividend_type: Type of dividend (for dividend events)
            payment_date: Date of payment (for dividend events)
            declared_date: Date the event was declared
            ex_dividend_date: Ex-dividend date (for dividend events)
            record_date: Record date (for dividend events)
            currency: Currency code (for dividend events)
            yield_value: Dividend yield (for dividend events)
            close_price: Close price on the event date
            figi: Financial Instrument Global Identifier
            isin: International Securities Identification Number
            regularity: Regularity of dividend payment
            effective_date: Effective date for the event
            announcement_date: Date the event was announced
            additional_properties: Additional properties as a dictionary
        """
        self.ticker = ticker
        self.event_date = event_date
        self.event_type = event_type
        self.event_value = event_value
        self.details = details
        self.source = source
        
        # Additional fields
        self.dividend_type = dividend_type
        self.payment_date = payment_date
        self.declared_date = declared_date
        self.ex_dividend_date = ex_dividend_date
        self.record_date = record_date
        self.currency = currency
        self.yield_value = yield_value
        self.close_price = close_price
        self.figi = figi
        self.isin = isin
        self.regularity = regularity
        self.effective_date = effective_date
        self.announcement_date = announcement_date
        self.additional_properties = additional_properties or {}
    
    @classmethod
    def from_entity(cls, entity: CorporateEvent) -> 'CorporateEventDTO':
        """
        Create a CorporateEventDTO from a CorporateEvent entity
        
        Args:
            entity: CorporateEvent domain entity
            
        Returns:
            CorporateEventDTO instance
        """
        # Convert the entity to a dict first
        entity_dict = entity.to_dict()
        
        # Extract basic fields
        basic_fields = {
            "ticker": entity.ticker,
            "event_date": entity.event_date.isoformat() if isinstance(entity.event_date, date) else entity.event_date,
            "event_type": entity.event_type.value if isinstance(entity.event_type, EventType) else entity.event_type,
            "event_value": entity.event_value,
            "details": entity.details,
            "source": entity.source,
            "figi": entity.figi,
            "isin": entity.isin,
        }
        
        # Add dividend-specific fields
        if entity.event_type == EventType.DIVIDEND:
            dividend_fields = {
                "dividend_type": entity.dividend_type.value if entity.dividend_type else None,
                "payment_date": entity.payment_date.isoformat() if entity.payment_date else None,
                "declared_date": entity.declared_date.isoformat() if entity.declared_date else None,
                "ex_dividend_date": entity.ex_dividend_date.isoformat() if entity.ex_dividend_date else None,
                "record_date": entity.record_date.isoformat() if entity.record_date else None,
                "currency": entity.currency,
                "yield_value": entity.yield_value,
                "close_price": entity.close_price,
                "regularity": entity.regularity,
            }
            basic_fields.update({k: v for k, v in dividend_fields.items() if v is not None})
        
        # Add other date fields
        date_fields = {
            "effective_date": entity.effective_date.isoformat() if entity.effective_date else None,
            "announcement_date": entity.announcement_date.isoformat() if entity.announcement_date else None,
        }
        basic_fields.update({k: v for k, v in date_fields.items() if v is not None})
        
        # Add additional properties
        basic_fields["additional_properties"] = entity.additional_properties
        
        return cls(**basic_fields)
    
    def to_entity(self) -> CorporateEvent:
        """
        Convert the DTO to a domain entity
        
        Returns:
            CorporateEvent domain entity
        """
        # Start with basic fields
        entity_data = {
            "ticker": self.ticker,
            "event_date": self.event_date,
            "event_type": self.event_type,
            "event_value": self.event_value,
            "details": self.details,
            "source": self.source,
            "figi": self.figi,
            "isin": self.isin,
        }
        
        # Add dividend-specific fields if this is a dividend event
        if self.event_type == "dividend":
            dividend_fields = {
                "dividend_type": self.dividend_type,
                "payment_date": self.payment_date,
                "declared_date": self.declared_date,
                "ex_dividend_date": self.ex_dividend_date,
                "record_date": self.record_date,
                "currency": self.currency,
                "yield_value": self.yield_value,
                "close_price": self.close_price,
                "regularity": self.regularity,
            }
            entity_data.update({k: v for k, v in dividend_fields.items() if v is not None})
        
        # Add other date fields
        date_fields = {
            "effective_date": self.effective_date,
            "announcement_date": self.announcement_date,
        }
        entity_data.update({k: v for k, v in date_fields.items() if v is not None})
        
        # Create the entity using from_dict to handle all conversions
        return CorporateEvent.from_dict(entity_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DTO to a dictionary
        
        Returns:
            Dictionary representation
        """
        # Start with basic fields
        result = {
            "ticker": self.ticker,
            "event_date": self.event_date,
            "event_type": self.event_type,
            "event_value": self.event_value,
            "details": self.details,
            "source": self.source,
        }
        
        # Add additional fields if they have values
        for field in [
            "dividend_type", "payment_date", "declared_date", "ex_dividend_date",
            "record_date", "currency", "yield_value", "close_price", "figi", "isin",
            "regularity", "effective_date", "announcement_date"
        ]:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value
                
        # Add additional properties
        if self.additional_properties:
            result.update(self.additional_properties)
            
        return result


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
    data = []
    for e in events:
        # Convert entity to dictionary to capture all fields
        event_dict = e.to_dict()
        
        # Ensure all dates are ISO format strings for pandas
        for key, value in event_dict.items():
            if isinstance(value, date) or isinstance(value, datetime):
                event_dict[key] = value.isoformat()
                
        data.append(event_dict)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date columns to datetime
    date_columns = [
        "event_date", "payment_date", "declared_date", "ex_dividend_date",
        "record_date", "effective_date", "announcement_date"
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df 