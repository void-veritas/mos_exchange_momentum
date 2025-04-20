from typing import List, Dict, Any, Optional, Union, Set
from datetime import date, datetime, timedelta

from ..entities.index import Index, IndexConstituent
from ..entities.security import Security
from ..interfaces.data_source import IndexDataSource


class IndexService:
    """
    Domain service for working with market indices
    """
    
    def __init__(self, data_sources: List[IndexDataSource]):
        """
        Initialize index service
        
        Args:
            data_sources: List of index data sources, in order of preference
        """
        self.data_sources = data_sources
    
    async def get_index_composition(self, 
                             index_id: str,
                             date_str: Optional[str] = None) -> Optional[Index]:
        """
        Get index composition for a specific date
        
        Args:
            index_id: Index identifier
            date_str: Date string in ISO format (YYYY-MM-DD)
            
        Returns:
            Index object if found, None otherwise
        """
        for source in self.data_sources:
            try:
                result = await source.fetch_index_composition(
                    index_id=index_id,
                    date_str=date_str
                )
                
                if result:
                    return result
            except Exception as e:
                # Log error and continue with next source
                print(f"Error fetching index from {source.name}: {e}")
                continue
        
        return None
    
    async def get_index_timeseries(self,
                            index_id: str,
                            start_date: Optional[Union[str, date]] = None,
                            end_date: Optional[Union[str, date]] = None,
                            frequency: str = "monthly") -> Dict[str, Index]:
        """
        Get index composition over time
        
        Args:
            index_id: Index identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency ("daily", "weekly", "monthly")
            
        Returns:
            Dictionary mapping dates (as strings) to Index objects
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = date.today()
        elif isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
            
        if start_date is None:
            start_date = end_date - timedelta(days=180)  # 6 months
        elif isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
            
        for source in self.data_sources:
            try:
                result = await source.fetch_index_timeseries(
                    index_id=index_id,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                
                if result:
                    return result
            except Exception as e:
                # Log error and continue with next source
                print(f"Error fetching index timeseries from {source.name}: {e}")
                continue
        
        return {}
    
    def analyze_index_changes(self, timeseries: Dict[str, Index]) -> Dict[str, Any]:
        """
        Analyze changes in index composition over time
        
        Args:
            timeseries: Dictionary mapping dates to Index objects
            
        Returns:
            Dictionary with analysis results
        """
        if not timeseries:
            return {
                "summary": {
                    "periods": 0,
                    "total_additions": 0,
                    "total_removals": 0,
                    "stocks_with_weight_changes": 0
                },
                "additions": {},
                "removals": {},
                "weight_changes": {}
            }
        
        # Sort dates
        dates = sorted(timeseries.keys())
        
        # Initialize results
        result = {
            "summary": {
                "periods": len(dates) - 1,
                "total_additions": 0,
                "total_removals": 0,
                "stocks_with_weight_changes": 0
            },
            "additions": {},
            "removals": {},
            "weight_changes": {}
        }
        
        # Analyze changes between consecutive dates
        for i in range(1, len(dates)):
            current_date = dates[i]
            previous_date = dates[i-1]
            
            current_index = timeseries[current_date]
            previous_index = timeseries[previous_date]
            
            # Get tickers in each index
            current_tickers = {constituent.security.ticker for constituent in current_index.constituents}
            previous_tickers = {constituent.security.ticker for constituent in previous_index.constituents}
            
            # Find additions and removals
            additions = current_tickers - previous_tickers
            removals = previous_tickers - current_tickers
            
            # Store additions
            if additions:
                added_constituents = [
                    {
                        "ticker": constituent.security.ticker,
                        "name": constituent.security.name,
                        "weight": constituent.weight
                    }
                    for constituent in current_index.constituents
                    if constituent.security.ticker in additions
                ]
                
                result["additions"][current_date] = added_constituents
                result["summary"]["total_additions"] += len(additions)
            
            # Store removals
            if removals:
                removed_constituents = [
                    {
                        "ticker": constituent.security.ticker,
                        "name": constituent.security.name,
                        "weight": constituent.weight
                    }
                    for constituent in previous_index.constituents
                    if constituent.security.ticker in removals
                ]
                
                result["removals"][current_date] = removed_constituents
                result["summary"]["total_removals"] += len(removals)
            
            # Find weight changes for securities present in both indices
            common_tickers = current_tickers.intersection(previous_tickers)
            weight_changes = []
            
            for ticker in common_tickers:
                # Find weights in each index
                current_weight = next(
                    (constituent.weight for constituent in current_index.constituents 
                     if constituent.security.ticker == ticker), 
                    0
                )
                
                previous_weight = next(
                    (constituent.weight for constituent in previous_index.constituents 
                     if constituent.security.ticker == ticker), 
                    0
                )
                
                # Check if weight changed significantly (more than 0.5 percentage points)
                if abs(current_weight - previous_weight) > 0.5:
                    security = next(
                        (constituent.security for constituent in current_index.constituents 
                         if constituent.security.ticker == ticker), 
                        None
                    )
                    
                    if security:
                        weight_changes.append({
                            "ticker": ticker,
                            "name": security.name,
                            "previous_weight": previous_weight,
                            "current_weight": current_weight,
                            "change": current_weight - previous_weight
                        })
            
            if weight_changes:
                result["weight_changes"][current_date] = weight_changes
                result["summary"]["stocks_with_weight_changes"] += len(weight_changes)
        
        return result 