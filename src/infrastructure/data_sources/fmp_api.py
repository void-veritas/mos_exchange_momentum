import aiohttp
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union

from ...domain.interfaces.data_source import DataSource
from ...domain.entities.price import Price
from ...domain.entities.security import Security
from ...application.config import Config


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FMPDataSource(DataSource):
    """
    Implementation of DataSource for Financial Modeling Prep API
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP data source
        
        Args:
            api_key: FMP API key (if not provided, will use from Config)
        """
        self.api_key = api_key or Config.FMP_API_KEY
        if not self.api_key:
            logger.warning("FMP API key not provided. FMP data source will not work.")
    
    @property
    def name(self) -> str:
        return "FMP"
    
    async def fetch_prices(self, 
                     tickers: List[str],
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None) -> Dict[str, List[Price]]:
        """
        Fetch price data for the specified tickers from FMP
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        if not self.api_key:
            logger.error("FMP API key not configured. Cannot fetch prices.")
            return {}
        
        # Format dates
        if end_date is None:
            end_date_dt = datetime.now()
            end_date_str = end_date_dt.strftime("%Y-%m-%d")
        elif isinstance(end_date, date):
            end_date_dt = datetime.combine(end_date, datetime.min.time())
            end_date_str = end_date.isoformat()
        else:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_str = end_date
            
        if start_date is None:
            # Default to 30 days ago
            start_date_dt = end_date_dt - timedelta(days=30)
            start_date_str = start_date_dt.strftime("%Y-%m-%d")
        elif isinstance(start_date, date):
            start_date_dt = datetime.combine(start_date, datetime.min.time())
            start_date_str = start_date.isoformat()
        else:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_date_str = start_date
            
        result = {}
        
        # Fetch data for each ticker
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                # Build the URL for FMP API
                url = f"{self.BASE_URL}/historical-price-full/{ticker}"
                
                params = {
                    "from": start_date_str,
                    "to": end_date_str,
                    "apikey": self.api_key
                }
                
                try:
                    logger.info(f"Fetching data for {ticker} from FMP")
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            logger.error(f"Error fetching FMP data for {ticker}: {response.status}")
                            continue
                        
                        # Parse the JSON response
                        data = await response.json()
                        
                        # Check if we have data
                        if "historical" not in data:
                            logger.warning(f"No historical data available for {ticker}")
                            continue
                        
                        # Extract historical data
                        historical = data["historical"]
                        
                        # Create Price objects
                        prices = []
                        for item in historical:
                            # Parse date
                            price_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
                            
                            # Create Price object
                            price = Price(
                                ticker=ticker,
                                date=price_date,
                                open=item.get("open"),
                                high=item.get("high"),
                                low=item.get("low"),
                                close=item.get("close"),
                                adjusted_close=item.get("adjClose"),
                                volume=item.get("volume"),
                                source=self.name
                            )
                            
                            prices.append(price)
                        
                        # Sort by date (ascending)
                        prices.sort(key=lambda p: p.date)
                        
                        result[ticker] = prices
                        logger.info(f"Fetched {len(prices)} price points for {ticker} from FMP")
                        
                except Exception as e:
                    logger.error(f"Error fetching FMP data for {ticker}: {e}")
                    continue
        
        return result
    
    async def fetch_security_info(self, ticker: str) -> Optional[Security]:
        """
        Fetch metadata for a specific security from FMP
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        if not self.api_key:
            logger.error("FMP API key not configured. Cannot fetch security info.")
            return None
        
        # Build the URL for company profile
        url = f"{self.BASE_URL}/profile/{ticker}"
        
        params = {
            "apikey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching security info for {ticker}: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Check if we have data
                    if not data or not isinstance(data, list) or len(data) == 0:
                        logger.warning(f"No profile data available for {ticker}")
                        return None
                    
                    # Extract company profile
                    profile = data[0]
                    
                    # Create Security object
                    security = Security(
                        ticker=ticker,
                        name=profile.get("companyName", ticker),
                        figi=None,  # FMP doesn't provide FIGI
                        isin=None,  # FMP doesn't provide ISIN
                        currency=profile.get("currency", "USD"),
                        sector=profile.get("sector"),
                        exchange=profile.get("exchangeShortName", "")
                    )
                    
                    return security
                    
        except Exception as e:
            logger.error(f"Error fetching security info for {ticker}: {e}")
            return None 