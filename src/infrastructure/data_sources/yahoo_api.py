import aiohttp
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union

from ...domain.interfaces.data_source import DataSource
from ...domain.entities.price import Price
from ...domain.entities.security import Security


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YahooFinanceDataSource(DataSource):
    """
    Implementation of DataSource for Yahoo Finance
    """
    
    # List of Russian stock tickers on MOEX
    RUSSIAN_TICKERS = [
        "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "SNGS", "SNGSP", "TATN", "TATNP",
        "MGNT", "MTSS", "VTBR", "ALRS", "CHMF", "NLMK", "PLZL", "YNDX", "POLY", "TCSG"
    ]
    
    @property
    def name(self) -> str:
        return "YAHOO"
    
    def _format_ticker(self, ticker: str) -> str:
        """
        Format ticker symbol for Yahoo Finance API
        
        Args:
            ticker: Original ticker symbol
            
        Returns:
            Formatted ticker symbol
        """
        # Add .ME suffix for Russian stocks
        if ticker in self.RUSSIAN_TICKERS:
            return f"{ticker}.ME"
        return ticker
    
    async def fetch_prices(self, 
                     tickers: List[str],
                     start_date: Optional[Union[str, date]] = None,
                     end_date: Optional[Union[str, date]] = None) -> Dict[str, List[Price]]:
        """
        Fetch price data for the specified tickers from Yahoo Finance
        
        Args:
            tickers: List of ticker symbols (may need .ME suffix for Russian stocks)
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        # Set default dates
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
        
        # Convert dates to Unix timestamps (seconds)
        period1 = int(start_date_dt.timestamp())
        period2 = int(end_date_dt.timestamp())
        
        result = {}
        
        # Fetch data for each ticker
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                # Format ticker for Yahoo Finance
                yahoo_ticker = self._format_ticker(ticker)
                
                # Build the URL for Yahoo Finance API
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
                
                params = {
                    "period1": str(period1),
                    "period2": str(period2),
                    "interval": "1d",
                    "events": "history",
                    "includeAdjustedClose": "true"
                }
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                try:
                    logger.info(f"Fetching data for {ticker} (Yahoo ticker: {yahoo_ticker})")
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Error fetching Yahoo Finance data for {yahoo_ticker}: {response.status}")
                            continue
                        
                        # Parse the JSON response
                        data = await response.json()
                        
                        # Check if we have data
                        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
                            logger.warning(f"No data available for {yahoo_ticker}")
                            continue
                        
                        # Extract the result
                        chart_result = data["chart"]["result"][0]
                        
                        # Extract metadata
                        meta = chart_result.get("meta", {})
                        
                        # Extract timestamps and pricing data
                        timestamps = chart_result.get("timestamp", [])
                        quote = chart_result.get("indicators", {}).get("quote", [{}])[0]
                        adjclose = chart_result.get("indicators", {}).get("adjclose", [{}])[0]
                        
                        # Create Price objects
                        prices = []
                        for i, ts in enumerate(timestamps):
                            price_date = datetime.fromtimestamp(ts).date()
                            
                            # Get price data for this timestamp
                            open_price = quote.get("open", [])[i] if i < len(quote.get("open", [])) else None
                            high_price = quote.get("high", [])[i] if i < len(quote.get("high", [])) else None
                            low_price = quote.get("low", [])[i] if i < len(quote.get("low", [])) else None
                            close_price = quote.get("close", [])[i] if i < len(quote.get("close", [])) else None
                            volume = quote.get("volume", [])[i] if i < len(quote.get("volume", [])) else None
                            adj_close = adjclose.get("adjclose", [])[i] if adjclose and i < len(adjclose.get("adjclose", [])) else close_price
                            
                            # Create Price object
                            price = Price(
                                ticker=ticker,
                                date=price_date,
                                open=open_price,
                                high=high_price,
                                low=low_price,
                                close=close_price,
                                adjusted_close=adj_close,
                                volume=volume,
                                source=self.name
                            )
                            
                            prices.append(price)
                        
                        result[ticker] = prices
                        logger.info(f"Fetched {len(prices)} price points for {ticker} from Yahoo Finance")
                        
                except Exception as e:
                    logger.error(f"Error fetching Yahoo Finance data for {yahoo_ticker}: {e}")
                    continue
        
        return result
    
    async def fetch_security_info(self, ticker: str) -> Optional[Security]:
        """
        Fetch metadata for a specific security from Yahoo Finance
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        # Format ticker for Yahoo Finance
        yahoo_ticker = self._format_ticker(ticker)
        
        url = f"https://query1.finance.yahoo.com/v7/finance/quote"
        params = {"symbols": yahoo_ticker}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching security info for {yahoo_ticker}: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Check if we have data
                    if "quoteResponse" not in data or "result" not in data["quoteResponse"] or not data["quoteResponse"]["result"]:
                        logger.warning(f"No security info available for {yahoo_ticker}")
                        return None
                    
                    # Extract security info
                    quote = data["quoteResponse"]["result"][0]
                    
                    # Create Security object
                    security = Security(
                        ticker=ticker,
                        name=quote.get("longName") or quote.get("shortName") or ticker,
                        figi=None,  # Yahoo Finance doesn't provide FIGI
                        isin=None,  # Yahoo Finance doesn't provide ISIN
                        currency=quote.get("currency", "USD"),
                        sector=quote.get("sector"),
                        exchange=quote.get("exchange", "YAHOO")
                    )
                    
                    return security
                    
        except Exception as e:
            logger.error(f"Error fetching security info for {yahoo_ticker}: {e}")
            return None 