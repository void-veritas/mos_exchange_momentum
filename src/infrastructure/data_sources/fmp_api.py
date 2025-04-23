import aiohttp
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union

from ...domain.interfaces.data_source import DataSource
from ...domain.entities.price import Price
from ...domain.entities.security import Security
from ...domain.entities.corporate_event import CorporateEvent, EventType, DividendType
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
    
    async def fetch_corporate_events(self,
                               tickers: List[str],
                               event_types: Optional[List[EventType]] = None,
                               start_date: Optional[Union[str, date]] = None,
                               end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Fetch corporate events data for the specified tickers from FMP
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
        if not self.api_key:
            logger.error("FMP API key not configured. Cannot fetch corporate events.")
            return {ticker: [] for ticker in tickers}
        
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
            # Default to 1 year ago
            start_date_dt = end_date_dt - timedelta(days=365)
            start_date_str = start_date_dt.strftime("%Y-%m-%d")
        elif isinstance(start_date, date):
            start_date_dt = datetime.combine(start_date, datetime.min.time())
            start_date_str = start_date.isoformat()
        else:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_date_str = start_date
        
        result = {}
        
        # Determine which events to fetch based on event_types
        fetch_dividends = not event_types or EventType.DIVIDEND in event_types
        fetch_splits = not event_types or EventType.STOCK_SPLIT in event_types
        
        # Process each ticker
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                events = []
                
                # Fetch dividends if needed
                if fetch_dividends:
                    try:
                        dividend_events = await self._fetch_dividends(
                            ticker=ticker,
                            session=session,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )
                        events.extend(dividend_events)
                    except Exception as e:
                        logger.error(f"Error fetching dividend data for {ticker}: {e}")
                
                # Fetch stock splits if needed
                if fetch_splits:
                    try:
                        split_events = await self._fetch_splits(
                            ticker=ticker,
                            session=session,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )
                        events.extend(split_events)
                    except Exception as e:
                        logger.error(f"Error fetching split data for {ticker}: {e}")
                
                result[ticker] = events
        
        return result
    
    async def _fetch_dividends(self, 
                         ticker: str, 
                         session: aiohttp.ClientSession,
                         start_date: str,
                         end_date: str) -> List[CorporateEvent]:
        """
        Fetch dividend events for a specific ticker from FMP
        
        Args:
            ticker: Ticker symbol
            session: aiohttp session to use
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of CorporateEvent objects
        """
        url = f"{self.BASE_URL}/historical-price-full/stock_dividend/{ticker}"
        
        params = {
            "from": start_date,
            "to": end_date,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching dividend data for {ticker} from FMP")
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error fetching FMP dividend data for {ticker}: {response.status}")
                    return []
                
                # Parse the JSON response
                data = await response.json()
                
                # Check if we have data
                if "historical" not in data:
                    logger.warning(f"No historical dividend data available for {ticker}")
                    return []
                
                # Extract historical data
                historical = data["historical"]
                
                # Create CorporateEvent objects
                events = []
                for item in historical:
                    # Parse dates
                    date_str = item.get("date")
                    if not date_str:
                        continue
                    
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    # Get dividend information
                    dividend = item.get("dividend")
                    if dividend is None:
                        continue
                    
                    # Create CorporateEvent object
                    event = CorporateEvent(
                        ticker=ticker,
                        event_date=event_date,
                        event_type=EventType.DIVIDEND,
                        event_value=dividend,
                        dividend_type=DividendType.REGULAR,  # Default to regular
                        details=f"Dividend of {dividend} per share",
                        source=self.name
                    )
                    
                    events.append(event)
                
                logger.info(f"Fetched {len(events)} dividend events for {ticker} from FMP")
                return events
                
        except Exception as e:
            logger.error(f"Error fetching dividend data for {ticker} from FMP: {e}")
            return []
    
    async def _fetch_splits(self, 
                      ticker: str, 
                      session: aiohttp.ClientSession,
                      start_date: str,
                      end_date: str) -> List[CorporateEvent]:
        """
        Fetch stock split events for a specific ticker from FMP
        
        Args:
            ticker: Ticker symbol
            session: aiohttp session to use
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of CorporateEvent objects
        """
        url = f"{self.BASE_URL}/historical-price-full/stock_split/{ticker}"
        
        params = {
            "from": start_date,
            "to": end_date,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching stock split data for {ticker} from FMP")
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error fetching FMP stock split data for {ticker}: {response.status}")
                    return []
                
                # Parse the JSON response
                data = await response.json()
                
                # Check if we have data
                if "historical" not in data:
                    logger.warning(f"No historical stock split data available for {ticker}")
                    return []
                
                # Extract historical data
                historical = data["historical"]
                
                # Create CorporateEvent objects
                events = []
                for item in historical:
                    # Parse dates
                    date_str = item.get("date")
                    if not date_str:
                        continue
                    
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    # Get split information
                    split_text = item.get("label")
                    if not split_text:
                        continue
                    
                    # Parse split ratio (e.g., "2:1" -> 2.0)
                    try:
                        numerator, denominator = split_text.split(":")
                        split_ratio = float(numerator) / float(denominator)
                    except (ValueError, ZeroDivisionError):
                        logger.warning(f"Could not parse split ratio from '{split_text}' for {ticker}")
                        continue
                    
                    # Create CorporateEvent object
                    event = CorporateEvent(
                        ticker=ticker,
                        event_date=event_date,
                        event_type=EventType.STOCK_SPLIT,
                        event_value=split_ratio,
                        details=f"Stock split {split_text}",
                        source=self.name
                    )
                    
                    events.append(event)
                
                logger.info(f"Fetched {len(events)} stock split events for {ticker} from FMP")
                return events
                
        except Exception as e:
            logger.error(f"Error fetching stock split data for {ticker} from FMP: {e}")
            return [] 