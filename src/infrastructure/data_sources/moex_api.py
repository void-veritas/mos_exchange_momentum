import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union

from ...domain.interfaces.data_source import IndexDataSource, DataSource
from ...domain.entities.index import Index, IndexConstituent
from ...domain.entities.security import Security
from ...domain.entities.corporate_event import CorporateEvent, EventType, DividendType
from ...domain.entities.price import Price


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MOEX API endpoints
MOEX_BASE_URL = "https://iss.moex.com/iss"
INDEX_SECURITIES_URL = f"{MOEX_BASE_URL}/engines/stock/markets/index/boards/TQTF/securities.json"
SECURITY_INFO_URL = f"{MOEX_BASE_URL}/securities"

# Default index ID
DEFAULT_INDEX_ID = "IMOEX"


class MoexIndexDataSource(IndexDataSource):
    """
    Implementation of IndexDataSource for Moscow Exchange
    """
    
    @property
    def name(self) -> str:
        return "MOEX"
    
    async def fetch_index_composition(self,
                               index_id: str = DEFAULT_INDEX_ID,
                               date_str: Optional[str] = None) -> Optional[Index]:
        """
        Fetch stocks in the MOEX index for a specific date.
        
        Args:
            index_id: The ID of the index (default: IMOEX - Moscow Exchange Index)
            date_str: The date to fetch data for in YYYY-MM-DD format (default: latest trading day)
        
        Returns:
            Index object if found, None otherwise
        """
        # Format the date parameter
        date_param = f"&date={date_str}" if date_str else ""
        
        # Build the URL
        url = f"{INDEX_SECURITIES_URL}?iss.meta=off&iss.only=securities&securities.columns=SECID,SECNAME,WEIGHT,TRADEDATE&index={index_id}{date_param}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching data: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    if "securities" not in data or "data" not in data["securities"]:
                        logger.warning("No securities data found in response")
                        return None
                    
                    # Extract the securities data
                    securities_data = data["securities"]["data"]
                    columns = data["securities"]["columns"]
                    
                    # Map column names to indices
                    col_indices = {col: idx for idx, col in enumerate(columns)}
                    
                    # Parse the securities data
                    constituents = []
                    actual_date = None
                    
                    for row in securities_data:
                        if not row:
                            continue
                        
                        ticker = row[col_indices.get("SECID", 0)]
                        name = row[col_indices.get("SECNAME", 1)]
                        weight = row[col_indices.get("WEIGHT", 2)]
                        trade_date = row[col_indices.get("TRADEDATE", 3)]
                        
                        # Store the actual date from the first row
                        if not actual_date and trade_date:
                            actual_date = trade_date
                        
                        # Fetch additional info for the security
                        security_info = await self._fetch_security_info(ticker, session)
                        
                        # Create Security object
                        security = Security(
                            ticker=ticker,
                            name=name,
                            figi=security_info.get("figi"),
                            isin=security_info.get("isin"),
                            currency=security_info.get("currency", "RUB"),
                            sector=security_info.get("sector"),
                            exchange="MOEX"
                        )
                        
                        # Create IndexConstituent
                        constituent = IndexConstituent(
                            security=security,
                            weight=weight,
                            date=datetime.strptime(actual_date, "%Y-%m-%d").date() if actual_date else date.today()
                        )
                        
                        constituents.append(constituent)
                    
                    if not constituents:
                        logger.warning(f"No constituents found for index {index_id}")
                        return None
                    
                    # Create Index object
                    index = Index(
                        index_id=index_id,
                        name=f"Moscow Exchange Index ({index_id})",
                        date=datetime.strptime(actual_date, "%Y-%m-%d").date() if actual_date else date.today(),
                        constituents=constituents
                    )
                    
                    logger.info(f"Fetched {len(constituents)} constituents for {index_id} index on {actual_date or 'latest'}")
                    return index
        
        except Exception as e:
            logger.error(f"Error fetching MOEX index stocks: {e}")
            return None
    
    async def fetch_index_timeseries(self,
                              index_id: str = DEFAULT_INDEX_ID,
                              start_date: Optional[Union[str, date]] = None,
                              end_date: Optional[Union[str, date]] = None,
                              frequency: str = "monthly") -> Dict[str, Index]:
        """
        Fetch index composition over time
        
        Args:
            index_id: Index identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency ("daily", "weekly", "monthly")
            
        Returns:
            Dictionary mapping dates (as strings) to Index objects
        """
        # Set default dates if not provided
        if not end_date:
            end_date = date.today()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if not start_date:
            # Default to 6 months ago if not specified
            start_date = end_date - timedelta(days=180)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Generate date points based on frequency
        dates = self._generate_dates(start_date, end_date, frequency)
        
        # Fetch data for each date
        results = {}
        for date_point in dates:
            date_str = date_point.strftime("%Y-%m-%d")
            index = await self.fetch_index_composition(
                index_id=index_id,
                date_str=date_str
            )
            
            if index:
                results[date_str] = index
        
        return results
    
    async def _fetch_security_info(self, ticker: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Fetch additional information for a security.
        
        Args:
            ticker: The ticker symbol of the security
            session: aiohttp session to use
        
        Returns:
            A dictionary with security information
        """
        url = f"{SECURITY_INFO_URL}/{ticker}.json?iss.meta=off&iss.only=description"
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Error fetching security info for {ticker}: {response.status}")
                    return {}
                
                data = await response.json()
                
                if "description" not in data or "data" not in data["description"]:
                    return {}
                
                description_data = data["description"]["data"]
                
                # Convert to dictionary
                info = {}
                for row in description_data:
                    if len(row) >= 2:
                        key = row[0].lower()
                        value = row[1]
                        
                        # Extract specific fields of interest
                        if key == "secid":
                            info["ticker"] = value
                        elif key == "name":
                            info["name"] = value
                        elif key == "isin":
                            info["isin"] = value
                        elif key == "figi":
                            info["figi"] = value
                        elif key == "currencyid":
                            info["currency"] = value
                        elif key == "sectorsecurities":
                            info["sector"] = value
                
                return info
        
        except Exception as e:
            logger.warning(f"Error fetching security info for {ticker}: {e}")
            return {}
    
    def _generate_dates(self, start_dt: date, end_dt: date, frequency: str) -> List[date]:
        """
        Generate a list of dates based on the specified frequency.
        
        Args:
            start_dt: Start date
            end_dt: End date
            frequency: Frequency string ("daily", "weekly", "monthly")
        
        Returns:
            List of dates
        """
        dates = []
        
        if frequency == "daily":
            # Daily frequency
            current = start_dt
            while current <= end_dt:
                dates.append(current)
                current = current + timedelta(days=1)
                
        elif frequency == "weekly":
            # Weekly frequency (each Monday)
            current = start_dt
            # Move to next Monday if not already Monday
            while current.weekday() != 0:  # 0 is Monday
                current = current + timedelta(days=1)
                
            # Add all Mondays
            while current <= end_dt:
                dates.append(current)
                current = current + timedelta(days=7)
                
        elif frequency == "monthly":
            # Monthly frequency (1st of each month)
            current_year = start_dt.year
            current_month = start_dt.month
            
            while (current_year < end_dt.year) or (current_year == end_dt.year and current_month <= end_dt.month):
                # Use the 1st of the month
                month_date = date(current_year, current_month, 1)
                
                # Only add if it's within range
                if month_date >= start_dt and month_date <= end_dt:
                    dates.append(month_date)
                
                # Move to next month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
        else:
            # Default to monthly if unknown frequency
            return self._generate_dates(start_dt, end_dt, "monthly")
        
        return dates 


class MOEXDataSource(DataSource):
    """
    Implementation of DataSource for Moscow Exchange (MOEX)
    """
    
    BASE_URL = "https://iss.moex.com/iss"
    
    @property
    def name(self) -> str:
        return "MOEX"
    
    async def fetch_prices(self, 
                     tickers: List[str],
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None) -> Dict[str, List[Price]]:
        """
        Fetch price data for the specified tickers from MOEX
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        # Implementation would go here
        # For now, return empty dict as placeholder
        return {ticker: [] for ticker in tickers}
    
    async def fetch_security_info(self, ticker: str) -> Optional[Security]:
        """
        Fetch metadata for a specific security from MOEX
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        # Implementation would go here
        # For now, return None as placeholder
        return None
    
    async def fetch_corporate_events(self,
                               tickers: List[str],
                               event_types: Optional[List[EventType]] = None,
                               start_date: Optional[Union[str, date]] = None,
                               end_date: Optional[Union[str, date]] = None) -> Dict[str, List[CorporateEvent]]:
        """
        Fetch corporate events data for the specified tickers from MOEX
        
        Args:
            tickers: List of ticker symbols
            event_types: Optional list of event types to filter by
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker symbols to lists of CorporateEvent objects
        """
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
        
        # Get sample data for different tickers
        sample_data = self._get_sample_corporate_events()
        
        # Process each ticker
        for ticker in tickers:
            events = []
            
            # Check if we have sample data for this ticker
            if ticker in sample_data:
                ticker_events = sample_data[ticker]
                
                # Filter by event type if needed
                if event_types:
                    ticker_events = [e for e in ticker_events if e["event_type"] in [et.value for et in event_types]]
                
                # Filter by date range
                for event_data in ticker_events:
                    event_date = datetime.strptime(event_data["event_date"], "%Y-%m-%d").date()
                    
                    # Check if event is within date range
                    if start_date_dt.date() <= event_date <= end_date_dt.date():
                        # Create event based on type
                        event_type = event_data["event_type"]
                        
                        if event_type == "dividend":
                            event = self._create_dividend_event(ticker, event_data)
                        elif event_type == "stock_split":
                            event = self._create_split_event(ticker, event_data)
                        elif event_type in ["merger", "acquisition"]:
                            event = self._create_ma_event(ticker, event_data)
                        elif event_type == "ticker_change":
                            event = self._create_ticker_change_event(ticker, event_data)
                        else:
                            # Generic event
                            event = CorporateEvent(
                                ticker=ticker,
                                event_date=event_date,
                                event_type=EventType(event_type),
                                event_value=event_data.get("event_value"),
                                details=event_data.get("details"),
                                source=self.name
                            )
                        
                        events.append(event)
            
            # Add events to result
            result[ticker] = events
            
        return result
    
    def _get_sample_corporate_events(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get sample corporate events data for testing
        
        Returns:
            Dictionary with sample events data
        """
        return {
            "SBER": [
                # Dividend
                {
                    "event_date": "2020-10-05",
                    "event_type": "dividend",
                    "event_value": 18.70,
                    "payment_date": "2020-10-05",
                    "declared_date": "2020-09-25",
                    "ex_dividend_date": "2020-10-02",
                    "record_date": "2020-10-02",
                    "currency": "RUB",
                    "yield_value": 8.20,
                    "close_price": 227.80,
                    "dividend_type": "annual",
                    "details": "Annual dividend payment for 2019 fiscal year"
                },
                # Earlier dividend
                {
                    "event_date": "2020-04-12",
                    "event_type": "dividend",
                    "event_value": 5.20,
                    "payment_date": "2020-04-20",
                    "declared_date": "2020-03-30",
                    "ex_dividend_date": "2020-04-12",
                    "record_date": "2020-04-14",
                    "currency": "RUB",
                    "yield_value": 2.30,
                    "close_price": 226.10,
                    "dividend_type": "interim",
                    "details": "Interim dividend payment for H2 2019"
                },
                # Name change (hypothetical for example)
                {
                    "event_date": "2020-05-01",
                    "event_type": "name_change",
                    "details": "Changed legal name from 'Sberbank of Russia' to 'Sberbank'",
                    "old_name": "Sberbank of Russia",
                    "new_name": "Sberbank"
                }
            ],
            "GAZP": [
                # Dividend
                {
                    "event_date": "2020-07-20",
                    "event_type": "dividend",
                    "event_value": 15.24,
                    "payment_date": "2020-07-20",
                    "declared_date": "2020-06-26",
                    "ex_dividend_date": "2020-07-17",
                    "record_date": "2020-07-17",
                    "currency": "RUB",
                    "yield_value": 6.40,
                    "close_price": 237.50,
                    "dividend_type": "annual",
                    "details": "Annual dividend payment for 2019 fiscal year"
                }
            ],
            "YNDX": [
                # Merger (hypothetical example)
                {
                    "event_date": "2020-12-10",
                    "event_type": "merger",
                    "details": "[HYPOTHETICAL] Merged with Taxi division to create new unified transportation platform",
                    "partner": "Yandex.Taxi",
                    "merger_ratio": 1.0,
                    "announcement_date": "2020-11-01"
                }
            ],
            "MGNT": [
                # Stock Split (hypothetical example)
                {
                    "event_date": "2020-03-08",
                    "event_type": "stock_split",
                    "event_value": 5.0,  # 5-for-1 split
                    "details": "[HYPOTHETICAL] 5-for-1 stock split to increase liquidity and accessibility",
                    "announcement_date": "2020-02-10",
                    "effective_date": "2020-03-08"
                },
                # Ticker change (hypothetical example)
                {
                    "event_date": "2020-09-01",
                    "event_type": "ticker_change",
                    "old_ticker": "MGNT",
                    "new_ticker": "MGNT1",
                    "details": "[HYPOTHETICAL] Ticker symbol changed due to corporate restructuring",
                    "effective_date": "2020-09-01"
                }
            ]
        }
    
    def _create_dividend_event(self, ticker: str, data: Dict[str, Any]) -> CorporateEvent:
        """
        Create a dividend event from data
        
        Args:
            ticker: Ticker symbol
            data: Dividend event data
            
        Returns:
            CorporateEvent object
        """
        return CorporateEvent(
            ticker=ticker,
            event_date=datetime.strptime(data["event_date"], "%Y-%m-%d").date(),
            event_type=EventType.DIVIDEND,
            event_value=data["event_value"],
            dividend_type=DividendType(data["dividend_type"]),
            payment_date=datetime.strptime(data["payment_date"], "%Y-%m-%d").date(),
            declared_date=datetime.strptime(data["declared_date"], "%Y-%m-%d").date(),
            ex_dividend_date=datetime.strptime(data["ex_dividend_date"], "%Y-%m-%d").date(),
            record_date=datetime.strptime(data["record_date"], "%Y-%m-%d").date(),
            currency=data["currency"],
            yield_value=data["yield_value"],
            close_price=data["close_price"],
            details=data["details"],
            source=self.name
        )
    
    def _create_split_event(self, ticker: str, data: Dict[str, Any]) -> CorporateEvent:
        """
        Create a stock split event from data
        
        Args:
            ticker: Ticker symbol
            data: Stock split event data
            
        Returns:
            CorporateEvent object
        """
        event = CorporateEvent(
            ticker=ticker,
            event_date=datetime.strptime(data["event_date"], "%Y-%m-%d").date(),
            event_type=EventType.STOCK_SPLIT,
            event_value=data["event_value"],
            details=data["details"],
            source=self.name
        )
        
        # Add additional fields if present
        if "announcement_date" in data:
            event.announcement_date = datetime.strptime(data["announcement_date"], "%Y-%m-%d").date()
        if "effective_date" in data:
            event.effective_date = datetime.strptime(data["effective_date"], "%Y-%m-%d").date()
            
        return event
    
    def _create_ma_event(self, ticker: str, data: Dict[str, Any]) -> CorporateEvent:
        """
        Create a merger or acquisition event from data
        
        Args:
            ticker: Ticker symbol
            data: M&A event data
            
        Returns:
            CorporateEvent object
        """
        event = CorporateEvent(
            ticker=ticker,
            event_date=datetime.strptime(data["event_date"], "%Y-%m-%d").date(),
            event_type=EventType(data["event_type"]),
            details=data["details"],
            source=self.name
        )
        
        # Add additional fields to additional_properties
        for key, value in data.items():
            if key not in ["event_date", "event_type", "details"]:
                event.additional_properties[key] = value
                
        return event
    
    def _create_ticker_change_event(self, ticker: str, data: Dict[str, Any]) -> CorporateEvent:
        """
        Create a ticker change event from data
        
        Args:
            ticker: Ticker symbol
            data: Ticker change event data
            
        Returns:
            CorporateEvent object
        """
        event = CorporateEvent(
            ticker=ticker,
            event_date=datetime.strptime(data["event_date"], "%Y-%m-%d").date(),
            event_type=EventType.TICKER_CHANGE,
            details=data["details"],
            source=self.name
        )
        
        # Add additional fields to additional_properties
        for key, value in data.items():
            if key not in ["event_date", "event_type", "details"]:
                event.additional_properties[key] = value
                
        return event 