import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union

from ...domain.interfaces.data_source import IndexDataSource
from ...domain.entities.index import Index, IndexConstituent
from ...domain.entities.security import Security


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