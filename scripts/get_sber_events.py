#!/usr/bin/env python3
import asyncio
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.application.cli.corporate_events_cli import get_corporate_events, get_dividends


async def main():
    """
    Test fetching corporate events for SBER from 2020
    """
    ticker = "SBER"
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    print(f"\n=== Getting all corporate events for {ticker} from {start_date} to {end_date} ===")
    events_result = await get_corporate_events(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=True  # Force refresh to ensure we get the latest data
    )
    
    if "event_data" in events_result and not events_result["event_data"].empty:
        print(f"\nFound {len(events_result['event_data'])} events:")
        print(events_result["event_data"].to_string())
    else:
        print(f"\nNo events found for {ticker} in 2020")
    
    print(f"\n=== Getting dividend events for {ticker} from {start_date} to {end_date} ===")
    dividend_result = await get_dividends(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=True
    )
    
    if "event_data" in dividend_result and not dividend_result["event_data"].empty:
        print(f"\nFound {len(dividend_result['event_data'])} dividend events:")
        print(dividend_result["event_data"].to_string())
    else:
        print(f"\nNo dividend events found for {ticker} in 2020")


if __name__ == "__main__":
    asyncio.run(main()) 