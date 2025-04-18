import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import ssl
import certifi


async def fetch_moex_index_stocks(date: Optional[str] = None, verify_ssl: bool = False) -> List[Dict[str, Any]]:
    """
    Asynchronously fetch stocks included in the Moscow Exchange Index (MOEX) as of a specific date.
    
    Args:
        date: Date in format YYYY-MM-DD. If None, uses current date.
        verify_ssl: Whether to verify SSL certificates (set to False to bypass certificate errors)
    
    Returns:
        List of dictionaries with stock information including FIGI codes
    """
    # Format date or use current date if none provided
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching MOEX index constituents for date: {date}")
    
    # Skip directly to the analytics endpoint which showed data in previous run
    return await fetch_moex_index_alternative(date, verify_ssl=verify_ssl)


async def fetch_moex_index_timeseries(start_date: str, end_date: Optional[str] = None, 
                                      frequency: str = "weekly", verify_ssl: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch MOEX index constituents for a range of dates, creating a time series.
    
    Args:
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD (defaults to current date if None)
        frequency: How often to sample dates ('daily', 'weekly', 'monthly')
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        Dictionary with dates as keys and lists of stock data as values
    """
    # Validate and format dates
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date is None:
            end_date_obj = datetime.now()
        else:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            if end_date_obj > datetime.now():
                print(f"Warning: End date {end_date} is in the future. Using current date instead.")
                end_date_obj = datetime.now()
                
        if start_date_obj > end_date_obj:
            print(f"Error: Start date {start_date} is after end date {end_date}.")
            return {}
        
    except ValueError as e:
        print(f"Error with date format: {e}. Dates should be in YYYY-MM-DD format.")
        return {}
    
    # Generate list of dates based on frequency
    dates_to_fetch = []
    current_date = start_date_obj
    
    if frequency == "daily":
        # For daily, we'll fetch every business day
        while current_date <= end_date_obj:
            # Skip weekends (5=Saturday, 6=Sunday)
            if current_date.weekday() < 5:
                dates_to_fetch.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
    elif frequency == "weekly":
        # For weekly, get every Monday
        while current_date <= end_date_obj:
            # Move to the next Monday if we're not already on one
            while current_date.weekday() != 0 and current_date <= end_date_obj:
                current_date += timedelta(days=1)
            
            if current_date <= end_date_obj:
                dates_to_fetch.append(current_date.strftime("%Y-%m-%d"))
                current_date += timedelta(days=7)  # Jump to next week
    elif frequency == "monthly":
        # For monthly, get first business day of each month
        while current_date <= end_date_obj:
            # Set to 1st of the month
            first_of_month = current_date.replace(day=1)
            
            # If it's a weekend, move to the next business day
            while first_of_month.weekday() >= 5:
                first_of_month += timedelta(days=1)
            
            dates_to_fetch.append(first_of_month.strftime("%Y-%m-%d"))
            
            # Move to next month
            if first_of_month.month == 12:
                current_date = first_of_month.replace(year=first_of_month.year + 1, month=1)
            else:
                current_date = first_of_month.replace(month=first_of_month.month + 1)
    else:
        print(f"Unsupported frequency: {frequency}. Using weekly instead.")
        return await fetch_moex_index_timeseries(start_date, end_date, "weekly", verify_ssl)
    
    print(f"Fetching MOEX index data for {len(dates_to_fetch)} dates between {start_date} and {end_date or datetime.now().strftime('%Y-%m-%d')} ({frequency} frequency).")
    
    # Setup SSL context if needed
    ssl_context = None
    if not verify_ssl:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    # Create shared session for all requests
    async with aiohttp.ClientSession(connector=connector) as session:
        # Use gather with tasks to fetch data for all dates concurrently
        tasks = []
        for date in dates_to_fetch:
            tasks.append(fetch_moex_index_stocks(date, verify_ssl))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results into a date-keyed dictionary
        timeseries_data = {}
        for i, date in enumerate(dates_to_fetch):
            if isinstance(results[i], Exception):
                print(f"Error fetching data for {date}: {results[i]}")
                timeseries_data[date] = []
            else:
                # Use the actual date from the result as the key
                # (might be different if original date had no data)
                actual_date = results[i][0]["date"] if results[i] else date
                timeseries_data[actual_date] = results[i]
        
        return timeseries_data


async def fetch_moex_index_alternative(date: str, session: Optional[aiohttp.ClientSession] = None, 
                                      verify_ssl: bool = False) -> List[Dict[str, Any]]:
    """Method to fetch MOEX index constituents for a specific date."""
    # This endpoint provides historical index composition data
    url = "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/IMOEX.json"
    
    # Convert date string to datetime for manipulation
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        
        # If the date is in the future, use the current date instead
        if date_obj > datetime.now():
            print(f"Warning: Date {date} is in the future. Using current date instead.")
            date = datetime.now().strftime("%Y-%m-%d")
            date_obj = datetime.now()
    except ValueError:
        print(f"Invalid date format: {date}. Using current date instead.")
        date = datetime.now().strftime("%Y-%m-%d")
        date_obj = datetime.now()
    
    # Format date for MOEX API
    moex_date = date_obj.strftime("%Y-%m-%d")
    
    params = {
        "iss.meta": "off",
        "limit": 200,
        "date": moex_date  # Add date parameter for historical data
    }
    
    # Setup SSL context if needed
    ssl_context = None
    if not verify_ssl:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    close_session = False
    if session is None:
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        session = aiohttp.ClientSession(connector=connector)
        close_session = True
    
    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                # Try with nearby dates if the exact date is not available
                print(f"Data not available for {moex_date}, trying nearby dates...")
                return await try_nearby_dates(date_obj, session, verify_ssl=verify_ssl)
            
            data = await response.json()
            
            # Check if we got valid data with securities
            if "analytics" not in data or not data["analytics"]["data"]:
                print(f"No data available for {moex_date}, trying nearby dates...")
                return await try_nearby_dates(date_obj, session, verify_ssl=verify_ssl)
            
            print(f"Successfully retrieved data for {moex_date}")
            
            # Process the data
            columns = data["analytics"]["columns"]
            securities = data["analytics"]["data"]
            
            df = pd.DataFrame(securities, columns=columns)
            
            print(f"Found {len(df)} securities in the MOEX index for {moex_date}.")
            
            result = []
            for _, security in df.iterrows():
                ticker = security["ticker"] if "ticker" in security else security.get("secids", "")
                name = security["shortnames"] if "shortnames" in security else ""
                weight = security["weight"] if "weight" in security else None
                
                if pd.isna(ticker) or not ticker:
                    continue
                    
                figi = await fetch_figi_by_ticker(ticker, session, verify_ssl)
                
                result.append({
                    "ticker": ticker,
                    "name": name,
                    "figi": figi,
                    "weight": weight,
                    "date": moex_date
                })
            
            return result
    except Exception as e:
        print(f"Error fetching index constituents for {moex_date}: {e}")
        # Try with nearby dates if there's an error
        return await try_nearby_dates(date_obj, session, verify_ssl=verify_ssl)
    finally:
        if close_session:
            await session.close()


async def try_nearby_dates(date_obj: datetime, session: aiohttp.ClientSession, 
                           verify_ssl: bool = False) -> List[Dict[str, Any]]:
    """Try to fetch data from nearby dates if the exact date is not available."""
    # Try previous dates first (up to 5 business days back)
    for i in range(1, 6):
        prev_date = date_obj - timedelta(days=i)
        # Skip weekends
        if prev_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            continue
            
        prev_date_str = prev_date.strftime("%Y-%m-%d")
        print(f"Trying previous date: {prev_date_str}")
        
        # Try the previous date
        url = "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/IMOEX.json"
        params = {
            "iss.meta": "off",
            "limit": 200,
            "date": prev_date_str
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    continue
                
                data = await response.json()
                
                # Check if we got valid data
                if "analytics" not in data or not data["analytics"]["data"]:
                    continue
                
                print(f"Found data for nearby date: {prev_date_str}")
                
                # Process the data
                columns = data["analytics"]["columns"]
                securities = data["analytics"]["data"]
                
                df = pd.DataFrame(securities, columns=columns)
                
                result = []
                for _, security in df.iterrows():
                    ticker = security["ticker"] if "ticker" in security else security.get("secids", "")
                    name = security["shortnames"] if "shortnames" in security else ""
                    weight = security["weight"] if "weight" in security else None
                    
                    if pd.isna(ticker) or not ticker:
                        continue
                        
                    figi = await fetch_figi_by_ticker(ticker, session, verify_ssl)
                    
                    result.append({
                        "ticker": ticker,
                        "name": name,
                        "figi": figi,
                        "weight": weight,
                        "date": prev_date_str  # Use the actual date we found data for
                    })
                
                return result
        except Exception:
            continue
    
    # If no data found in previous dates, return empty list
    print("No data available for requested date or nearby dates.")
    return []


async def fetch_figi_by_ticker(ticker: str, session: aiohttp.ClientSession, 
                               verify_ssl: bool = False) -> str:
    """
    Fetch FIGI code for a specific ticker using MOEX API.
    
    Args:
        ticker: Stock ticker symbol
        session: Existing aiohttp client session
        verify_ssl: Whether to verify SSL certificates
    
    Returns:
        FIGI code as string or a MOEX-prefixed ID if not found
    """
    url = f"https://iss.moex.com/iss/securities/{ticker}.json"
    
    params = {
        "iss.meta": "off"
    }
    
    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return f"MOEX-{ticker}"  # Return default if API fails
            
            data = await response.json()
            
            # MOEX doesn't provide FIGI directly, but we can use ISIN as an alternative
            if "description" in data:
                columns = data["description"]["columns"]
                values = data["description"]["data"]
                
                df = pd.DataFrame(values, columns=columns)
                
                # Look for ISIN or similar identifier
                for identifier in ["ISIN", "SECID", "REGNUM"]:
                    id_row = df[df["name"] == identifier]
                    if not id_row.empty:
                        return id_row.iloc[0]["value"]  # Return identifier as alternative to FIGI
    except Exception:
        pass
        
    return f"MOEX-{ticker}"  # Return a placeholder if no identifier found


def get_index_changes(timeseries_data: Dict[str, List[Dict[str, Any]]]):
    """
    Analyze changes in index composition over time.
    
    Args:
        timeseries_data: Dictionary of date-keyed index constituent data
        
    Returns:
        Dictionary with analysis of changes (additions, removals, weight changes)
    """
    if not timeseries_data:
        return {
            "error": "No data available for analysis"
        }
    
    # Sort dates chronologically
    dates = sorted(timeseries_data.keys())
    
    if len(dates) < 2:
        return {
            "dates": dates,
            "message": "Need at least two dates to analyze changes"
        }
    
    changes = {
        "dates": dates,
        "additions": {},
        "removals": {},
        "weight_changes": {},
        "summary": {}
    }
    
    # Track constituents over time
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        prev_stocks = timeseries_data[prev_date]
        curr_stocks = timeseries_data[curr_date]
        
        prev_tickers = {s["ticker"] for s in prev_stocks}
        curr_tickers = {s["ticker"] for s in curr_stocks}
        
        # Find additions and removals
        added = curr_tickers - prev_tickers
        removed = prev_tickers - curr_tickers
        
        # Store changes
        if added:
            changes["additions"][curr_date] = [s for s in curr_stocks if s["ticker"] in added]
        
        if removed:
            changes["removals"][curr_date] = [s for s in prev_stocks if s["ticker"] in removed]
        
        # Track weight changes for stocks present in both periods
        for curr_stock in curr_stocks:
            if curr_stock["ticker"] not in added:
                # Find the same stock in previous period
                prev_stock = next((s for s in prev_stocks if s["ticker"] == curr_stock["ticker"]), None)
                if prev_stock and prev_stock["weight"] is not None and curr_stock["weight"] is not None:
                    weight_change = float(curr_stock["weight"]) - float(prev_stock["weight"])
                    if abs(weight_change) >= 0.1:  # Only track significant changes (>=0.1%)
                        if curr_stock["ticker"] not in changes["weight_changes"]:
                            changes["weight_changes"][curr_stock["ticker"]] = []
                        
                        changes["weight_changes"][curr_stock["ticker"]].append({
                            "from_date": prev_date,
                            "to_date": curr_date,
                            "from_weight": prev_stock["weight"],
                            "to_weight": curr_stock["weight"],
                            "change": weight_change
                        })
    
    # Create summary
    changes["summary"] = {
        "periods": len(dates) - 1,
        "total_additions": sum(len(stocks) for stocks in changes["additions"].values()),
        "total_removals": sum(len(stocks) for stocks in changes["removals"].values()),
        "stocks_with_weight_changes": len(changes["weight_changes"])
    }
    
    return changes 