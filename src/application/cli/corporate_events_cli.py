import asyncio
import argparse
import pandas as pd
from datetime import date
from typing import List, Optional

from ...domain.entities.corporate_event import EventType, DividendType
from ...domain.services.corporate_event_service import CorporateEventService
from ...application.services.corporate_event_fetcher import CorporateEventFetcherService
from ...infrastructure.repositories.corporate_event_repository import InMemoryCorporateEventRepository
from ...infrastructure.data_sources.moex_api import MOEXDataSource
from ...infrastructure.data_sources.fmp_api import FMPDataSource


async def get_corporate_events(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    force_refresh: bool = False,
    as_dataframe: bool = True
):
    """
    Get corporate events for the specified tickers
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        event_types: List of event types to filter by
        force_refresh: Whether to force fetching from data sources
        as_dataframe: Whether to return events as a pandas DataFrame
        
    Returns:
        Corporate events data
    """
    # Initialize repository
    repository = InMemoryCorporateEventRepository()
    
    # Initialize data sources
    data_sources = [
        MOEXDataSource(),  # MOEX API for Russian tickers
        FMPDataSource()    # FMP API for US tickers
    ]
    
    # Initialize services
    event_service = CorporateEventService(repository, data_sources)
    event_fetcher = CorporateEventFetcherService(event_service)
    
    # Get corporate events
    result = await event_fetcher.get_events(
        tickers=tickers,
        event_types=event_types,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
        as_dataframe=as_dataframe
    )
    
    return result


async def get_dividends(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dividend_types: Optional[List[str]] = None,
    force_refresh: bool = False,
    as_dataframe: bool = True
):
    """
    Get dividend events for the specified tickers
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dividend_types: List of dividend types to filter by
        force_refresh: Whether to force fetching from data sources
        as_dataframe: Whether to return events as a pandas DataFrame
        
    Returns:
        Dividend events data
    """
    # Initialize repository
    repository = InMemoryCorporateEventRepository()
    
    # Initialize data sources
    data_sources = [
        MOEXDataSource(),  # MOEX API for Russian tickers
        FMPDataSource()    # FMP API for US tickers
    ]
    
    # Initialize services
    event_service = CorporateEventService(repository, data_sources)
    event_fetcher = CorporateEventFetcherService(event_service)
    
    # Get dividend events
    result = await event_fetcher.get_dividends(
        tickers=tickers,
        dividend_types=dividend_types,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
        as_dataframe=as_dataframe
    )
    
    return result


def main():
    """
    Command-line interface for corporate events
    """
    parser = argparse.ArgumentParser(description="Fetch corporate events data")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--event-types", nargs="+", help="Event types to filter by")
    parser.add_argument("--dividend-types", nargs="+", help="Dividend types to filter by")
    parser.add_argument("--force-refresh", action="store_true", help="Force fetching from data sources")
    parser.add_argument("--dividends-only", action="store_true", help="Get dividend events only")
    
    args = parser.parse_args()
    
    # Run the appropriate function
    if args.dividends_only:
        result = asyncio.run(get_dividends(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            dividend_types=args.dividend_types,
            force_refresh=args.force_refresh
        ))
    else:
        result = asyncio.run(get_corporate_events(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            event_types=args.event_types,
            force_refresh=args.force_refresh
        ))
    
    # Display results
    if "event_data" in result and isinstance(result["event_data"], pd.DataFrame):
        if not result["event_data"].empty:
            print(f"\nFound {len(result['event_data'])} events:")
            print(result["event_data"].to_string())
        else:
            print(f"\nNo events found for the specified criteria.")
    else:
        for ticker, events in result.items():
            print(f"\n{ticker}: {len(events)} events")
            for event in events:
                print(f"  - {event.to_dict()}")


if __name__ == "__main__":
    main() 