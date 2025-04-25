import json
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import date, datetime

from src.domain.interfaces.price_repository import PriceRepository as PriceRepositoryInterface
from src.domain.entities.price import Price

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PriceRepository(PriceRepositoryInterface):
    """
    Implementation of the PriceRepository interface for storing and retrieving price data
    
    This implementation uses JSON files for persistence.
    """
    
    def __init__(self, storage_path: str = "data"):
        """
        Initialize the price repository
        
        Args:
            storage_path: Path to the directory where price data will be stored
        """
        self.storage_path = storage_path
        
        # Create the storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
    
    async def get_prices(self, 
                    tickers: List[str],
                    start_date: Optional[Union[str, date]] = None,
                    end_date: Optional[Union[str, date]] = None,
                    source: Optional[str] = None) -> Dict[str, List[Price]]:
        """
        Get price data for the specified tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        result = {}
        
        # Convert date strings to date objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Load data for each ticker
        for ticker in tickers:
            prices = []
            
            # Load from storage if exists
            storage_file = os.path.join(self.storage_path, f"{ticker.lower()}_prices.json")
            if os.path.exists(storage_file):
                try:
                    with open(storage_file, 'r') as f:
                        data = json.load(f)
                        
                    for price_data in data:
                        # Convert date string to date object for comparison
                        price_date = datetime.strptime(price_data["date"], "%Y-%m-%d").date() if isinstance(price_data["date"], str) else price_data["date"]
                        
                        # Filter by date range if specified
                        if start_date and price_date < start_date:
                            continue
                        if end_date and price_date > end_date:
                            continue
                            
                        # Filter by source if specified
                        if source and price_data.get("source") != source:
                            continue
                            
                        # Create Price object
                        prices.append(Price.from_dict(price_data))
                        
                    logger.info(f"Loaded {len(prices)} prices for {ticker} from {storage_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading prices for {ticker}: {e}")
            
            result[ticker] = prices
            
        return result
    
    async def store_prices(self, prices: List[Price]) -> None:
        """
        Store price data
        
        Args:
            prices: List of Price objects to save
        """
        # Group prices by ticker
        prices_by_ticker = {}
        for price in prices:
            if price.ticker not in prices_by_ticker:
                prices_by_ticker[price.ticker] = []
            prices_by_ticker[price.ticker].append(price)
        
        # Update storage files
        for ticker, ticker_prices in prices_by_ticker.items():
            # Format ticker correctly for storage
            ticker_formatted = ticker.lower()
            
            # Load existing data if available
            storage_file = os.path.join(self.storage_path, f"{ticker_formatted}_prices.json")
            existing_data = []
            
            if os.path.exists(storage_file):
                try:
                    with open(storage_file, 'r') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading existing data for {ticker}: {e}")
            
            # Create lookup of existing data by date
            existing_by_date = {item["date"]: item for item in existing_data}
            
            # Update or add new prices
            for price in ticker_prices:
                price_dict = price.to_dict()
                date_str = price_dict["date"]
                
                if date_str in existing_by_date:
                    # Update existing entry
                    existing_by_date[date_str].update(price_dict)
                else:
                    # Add new entry
                    existing_data.append(price_dict)
            
            # Sort by date
            existing_data.sort(key=lambda x: x["date"])
            
            # Save back to storage
            try:
                with open(storage_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                    
                logger.info(f"Saved {len(ticker_prices)} prices to {storage_file}")
                
            except Exception as e:
                logger.error(f"Error saving prices for {ticker}: {e}")
        
        return len(prices)
    
    async def get_latest_prices(self, 
                          tickers: List[str],
                          source: Optional[str] = None) -> Dict[str, Optional[Price]]:
        """
        Get the latest price for each of the specified tickers
        
        Args:
            tickers: List of ticker symbols
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to the latest Price object, or None if not found
        """
        result = {}
        
        # Load data for each ticker
        for ticker in tickers:
            latest_price = None
            
            # Load from storage if exists
            storage_file = os.path.join(self.storage_path, f"{ticker.lower()}_prices.json")
            if os.path.exists(storage_file):
                try:
                    with open(storage_file, 'r') as f:
                        data = json.load(f)
                        
                    if data:
                        # Filter by source if specified
                        if source:
                            data = [p for p in data if p.get("source") == source]
                            
                        if data:
                            # Sort by date (in case it's not already sorted)
                            data.sort(key=lambda x: x["date"], reverse=True)
                            
                            # Get the most recent price
                            latest_price = Price.from_dict(data[0])
                        
                except Exception as e:
                    logger.error(f"Error loading latest price for {ticker}: {e}")
            
            result[ticker] = latest_price
            
        return result
    
    async def delete_prices(self,
                      tickers: List[str],
                      start_date: Optional[Union[str, date]] = None,
                      end_date: Optional[Union[str, date]] = None,
                      source: Optional[str] = None) -> int:
        """
        Delete price entries matching the criteria
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            source: Optional source to filter by
            
        Returns:
            Number of deleted entries
        """
        total_deleted = 0
        
        # Convert date strings to date objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Process each ticker
        for ticker in tickers:
            storage_file = os.path.join(self.storage_path, f"{ticker.lower()}_prices.json")
            
            if os.path.exists(storage_file):
                try:
                    # Load existing data
                    with open(storage_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    original_count = len(existing_data)
                    
                    # Filter data based on criteria
                    filtered_data = []
                    for price_data in existing_data:
                        # Convert date string to date object for comparison
                        price_date = datetime.strptime(price_data["date"], "%Y-%m-%d").date() if isinstance(price_data["date"], str) else price_data["date"]
                        
                        # Check if we should keep this entry
                        keep = True
                        
                        # Filter by date range if specified
                        if start_date and price_date < start_date:
                            keep = False
                        if end_date and price_date > end_date:
                            keep = False
                            
                        # Filter by source if specified
                        if source and price_data.get("source") != source:
                            keep = False
                            
                        if keep:
                            filtered_data.append(price_data)
                    
                    # Calculate how many items were deleted
                    deleted_count = original_count - len(filtered_data)
                    total_deleted += deleted_count
                    
                    if deleted_count > 0:
                        # Save filtered data back to file
                        with open(storage_file, 'w') as f:
                            json.dump(filtered_data, f, indent=2)
                            
                        logger.info(f"Deleted {deleted_count} prices from {storage_file}")
                    
                except Exception as e:
                    logger.error(f"Error deleting prices for {ticker}: {e}")
        
        return total_deleted 