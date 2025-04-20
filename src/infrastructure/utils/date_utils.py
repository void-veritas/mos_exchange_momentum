from datetime import datetime, date, timedelta
from typing import List, Optional, Union, Tuple


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse a date string in various formats
    
    Args:
        date_str: Date string to parse (YYYY-MM-DD or other ISO format)
        
    Returns:
        Parsed date object or None if parsing fails
    """
    if not date_str:
        return None
        
    try:
        # Try ISO format first
        return date.fromisoformat(date_str)
    except ValueError:
        pass
        
    # Try other formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
            
    return None


def format_date(date_obj: Optional[Union[date, datetime]]) -> Optional[str]:
    """
    Format a date object to ISO string
    
    Args:
        date_obj: Date or datetime object
        
    Returns:
        ISO formatted date string (YYYY-MM-DD)
    """
    if not date_obj:
        return None
        
    if isinstance(date_obj, datetime):
        date_obj = date_obj.date()
        
    return date_obj.isoformat()


def get_date_range(start_date: Optional[Union[str, date, datetime]] = None,
                   end_date: Optional[Union[str, date, datetime]] = None,
                   days: int = 30) -> Tuple[date, date]:
    """
    Get a date range with proper defaults
    
    Args:
        start_date: Start date (optional)
        end_date: End date (optional)
        days: Number of days to look back if start_date is not provided
        
    Returns:
        Tuple of (start_date, end_date) as date objects
    """
    # Set end date
    if end_date is None:
        end = date.today()
    elif isinstance(end_date, str):
        end = parse_date(end_date) or date.today()
    elif isinstance(end_date, datetime):
        end = end_date.date()
    else:
        end = end_date
        
    # Set start date
    if start_date is None:
        start = end - timedelta(days=days)
    elif isinstance(start_date, str):
        start = parse_date(start_date) or (end - timedelta(days=days))
    elif isinstance(start_date, datetime):
        start = start_date.date()
    else:
        start = start_date
        
    return start, end


def generate_date_points(start_date: date, end_date: date, frequency: str = "daily") -> List[date]:
    """
    Generate date points between start and end dates based on frequency
    
    Args:
        start_date: Start date
        end_date: End date
        frequency: Frequency - "daily", "weekly", "monthly"
        
    Returns:
        List of dates
    """
    dates = []
    
    if frequency == "daily":
        # Daily dates
        current = start_date
        while current <= end_date:
            dates.append(current)
            current = current + timedelta(days=1)
            
    elif frequency == "weekly":
        # Weekly dates (each Monday)
        current = start_date
        # Move to next Monday if not already Monday
        while current.weekday() != 0:  # 0 is Monday
            current = current + timedelta(days=1)
            
        # Add all Mondays
        while current <= end_date:
            dates.append(current)
            current = current + timedelta(days=7)
            
    elif frequency == "monthly":
        # Monthly dates (1st of each month)
        current_year = start_date.year
        current_month = start_date.month
        
        while (current_year < end_date.year) or (current_year == end_date.year and current_month <= end_date.month):
            # Use the 1st of the month
            month_date = date(current_year, current_month, 1)
            
            # Only add if it's within range
            if month_date >= start_date and month_date <= end_date:
                dates.append(month_date)
            
            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
    else:
        # Default to daily if unknown frequency
        return generate_date_points(start_date, end_date, "daily")
    
    return dates 