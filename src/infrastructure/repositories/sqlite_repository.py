import sqlite3
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
import json

from ...domain.interfaces.price_repository import PriceRepository, SecurityRepository
from ...domain.entities.price import Price
from ...domain.entities.security import Security


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLitePriceRepository(PriceRepository, SecurityRepository):
    """
    SQLite implementation of PriceRepository and SecurityRepository
    """
    
    def __init__(self, db_path: str = 'moex_prices.db'):
        """
        Initialize SQLite repository
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """
        Initialize database schema if it doesn't exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS securities (
            id INTEGER PRIMARY KEY,
            ticker TEXT UNIQUE,
            name TEXT,
            figi TEXT,
            isin TEXT,
            currency TEXT,
            sector TEXT,
            exchange TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume INTEGER,
            source TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date, source)
        )
        ''')
        
        # Insert default data sources
        sources = [
            (1, 'MOEX', 'Moscow Exchange API'),
            (2, 'FMP', 'Financial Modeling Prep API'),
            (3, 'YAHOO', 'Yahoo Finance API'),
            (4, 'FINAM', 'Finam Export data'),
            (5, 'TINKOFF', 'Tinkoff Invest API'),
            (6, 'INVESTING', 'Investing.com data')
        ]
        
        cursor.executemany(
            'INSERT OR IGNORE INTO data_sources (id, name, description) VALUES (?, ?, ?)',
            sources
        )
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_date ON price_data (ticker, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON price_data (date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON securities (ticker)')
        
        conn.commit()
        conn.close()
    
    async def store_prices(self, prices: List[Price]) -> None:
        """
        Store prices in the database
        
        Args:
            prices: List of Price objects to store
        """
        # Use run_in_executor to run synchronous SQLite in async context
        def _store_prices_sync(prices):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                for price in prices:
                    # Convert to dictionary
                    data = price.to_dict()
                    
                    # Insert into database
                    cursor.execute(
                        '''
                        INSERT OR REPLACE INTO price_data 
                        (ticker, date, open, high, low, close, adjusted_close, volume, source, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            data["ticker"],
                            data["date"],
                            data["open"],
                            data["high"],
                            data["low"],
                            data["close"],
                            data["adjusted_close"],
                            data["volume"],
                            data["source"],
                            datetime.now().isoformat()
                        )
                    )
                
                conn.commit()
                logger.info(f"Stored {len(prices)} price records in SQLite")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error storing prices in SQLite: {e}")
                raise
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _store_prices_sync, prices)
    
    async def get_prices(self, 
                   tickers: List[str], 
                   start_date: Optional[Union[str, date]] = None,
                   end_date: Optional[Union[str, date]] = None,
                   source: Optional[str] = None) -> Dict[str, List[Price]]:
        """
        Retrieve prices for the specified tickers and date range
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to lists of Price objects
        """
        # Format dates
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
            
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
            
        # Build query function for thread pool
        def _get_prices_sync(tickers, start_date, end_date, source):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            result = {}
            
            try:
                for ticker in tickers:
                    # Build query
                    query = "SELECT * FROM price_data WHERE ticker = ?"
                    params = [ticker]
                    
                    if start_date:
                        query += " AND date >= ?"
                        params.append(start_date)
                        
                    if end_date:
                        query += " AND date <= ?"
                        params.append(end_date)
                        
                    if source:
                        query += " AND source = ?"
                        params.append(source)
                        
                    query += " ORDER BY date"
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to Price objects
                    prices = []
                    for row in rows:
                        # Convert row to dict
                        data = dict(row)
                        price = Price.from_dict(data)
                        prices.append(price)
                    
                    if prices:
                        result[ticker] = prices
                
                return result
                
            except Exception as e:
                logger.error(f"Error retrieving prices from SQLite: {e}")
                return {}
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _get_prices_sync, tickers, start_date, end_date, source
        )
    
    async def get_latest_prices(self, 
                         tickers: List[str],
                         source: Optional[str] = None) -> Dict[str, Price]:
        """
        Get the latest price for each ticker
        
        Args:
            tickers: List of ticker symbols
            source: Optional source to filter by
            
        Returns:
            Dictionary mapping ticker symbols to their latest Price object
        """
        def _get_latest_prices_sync(tickers, source):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            result = {}
            
            try:
                for ticker in tickers:
                    # Build query
                    query = """
                    SELECT * FROM price_data 
                    WHERE ticker = ?
                    """
                    params = [ticker]
                    
                    if source:
                        query += " AND source = ?"
                        params.append(source)
                        
                    query += " ORDER BY date DESC LIMIT 1"
                    
                    # Execute query
                    cursor.execute(query, params)
                    row = cursor.fetchone()
                    
                    if row:
                        # Convert to Price object
                        data = dict(row)
                        price = Price.from_dict(data)
                        result[ticker] = price
                
                return result
                
            except Exception as e:
                logger.error(f"Error retrieving latest prices from SQLite: {e}")
                return {}
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_latest_prices_sync, tickers, source)
    
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
        # Format dates
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
            
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
            
        def _delete_prices_sync(tickers, start_date, end_date, source):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Build query
                placeholders = ','.join('?' * len(tickers))
                query = f"DELETE FROM price_data WHERE ticker IN ({placeholders})"
                params = tickers.copy()
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                    
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                # Execute query
                cursor.execute(query, params)
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Deleted {deleted_count} price records from SQLite")
                return deleted_count
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error deleting prices from SQLite: {e}")
                return 0
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _delete_prices_sync, tickers, start_date, end_date, source
        )
    
    async def store_securities(self, securities: List[Security]) -> None:
        """
        Store securities in the database
        
        Args:
            securities: List of Security objects to store
        """
        def _store_securities_sync(securities):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                for security in securities:
                    # Convert to dictionary
                    data = security.to_dict()
                    
                    # Insert into database
                    cursor.execute(
                        '''
                        INSERT OR REPLACE INTO securities 
                        (ticker, name, figi, isin, currency, sector, exchange, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            data["ticker"],
                            data["name"],
                            data["figi"],
                            data["isin"],
                            data["currency"],
                            data["sector"],
                            data["exchange"],
                            datetime.now().isoformat()
                        )
                    )
                
                conn.commit()
                logger.info(f"Stored {len(securities)} securities in SQLite")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error storing securities in SQLite: {e}")
                raise
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _store_securities_sync, securities)
    
    async def get_securities(self, 
                      tickers: Optional[List[str]] = None) -> List[Security]:
        """
        Retrieve securities matching the criteria
        
        Args:
            tickers: Optional list of ticker symbols to filter by
            
        Returns:
            List of Security objects
        """
        def _get_securities_sync(tickers):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                if tickers:
                    # Build query with placeholders
                    placeholders = ','.join('?' * len(tickers))
                    query = f"SELECT * FROM securities WHERE ticker IN ({placeholders})"
                    cursor.execute(query, tickers)
                else:
                    # Get all securities
                    query = "SELECT * FROM securities"
                    cursor.execute(query)
                
                rows = cursor.fetchall()
                
                # Convert to Security objects
                securities = []
                for row in rows:
                    data = dict(row)
                    security = Security.from_dict(data)
                    securities.append(security)
                
                return securities
                
            except Exception as e:
                logger.error(f"Error retrieving securities from SQLite: {e}")
                return []
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_securities_sync, tickers)
    
    async def get_security(self, ticker: str) -> Optional[Security]:
        """
        Get a single security by ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Security object if found, None otherwise
        """
        def _get_security_sync(ticker):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                # Execute query
                cursor.execute("SELECT * FROM securities WHERE ticker = ?", (ticker,))
                row = cursor.fetchone()
                
                if row:
                    # Convert to Security object
                    data = dict(row)
                    return Security.from_dict(data)
                
                return None
                
            except Exception as e:
                logger.error(f"Error retrieving security from SQLite: {e}")
                return None
                
            finally:
                conn.close()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_security_sync, ticker) 