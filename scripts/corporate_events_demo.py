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
    Demo script to test fetching various corporate events for multiple tickers
    """
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    force_refresh = True  # Force refresh to ensure we get the latest data
    
    # Test SBER
    print("\n=== TESTING SBER ===")
    await test_ticker("SBER", start_date, end_date, force_refresh)
    
    # Test GAZP
    print("\n=== TESTING GAZP ===")
    await test_ticker("GAZP", start_date, end_date, force_refresh)
    
    # Test YNDX
    print("\n=== TESTING YNDX ===")
    await test_ticker("YNDX", start_date, end_date, force_refresh)
    
    # Test MGNT
    print("\n=== TESTING MGNT ===")
    await test_ticker("MGNT", start_date, end_date, force_refresh)
    
    # Test specific event types
    print("\n=== TESTING STOCK SPLITS ===")
    await test_event_type(["SBER", "MGNT"], "stock_split", start_date, end_date, force_refresh)
    
    print("\n=== TESTING MERGERS AND ACQUISITIONS ===")
    await test_event_type(["YNDX"], "merger", start_date, end_date, force_refresh)
    
    print("\n=== TESTING TICKER CHANGES ===")
    await test_event_type(["MGNT"], "ticker_change", start_date, end_date, force_refresh)
    
    print("\n=== TESTING NAME CHANGES ===")
    await test_event_type(["SBER"], "name_change", start_date, end_date, force_refresh)


async def test_ticker(ticker, start_date, end_date, force_refresh):
    """Test all events for a specific ticker"""
    print(f"\n=== Getting all corporate events for {ticker} from {start_date} to {end_date} ===")
    events_result = await get_corporate_events(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh
    )
    
    if "event_data" in events_result and not events_result["event_data"].empty:
        print(f"\nFound {len(events_result['event_data'])} events:")
        print(events_result["event_data"][["ticker", "event_date", "event_type", "event_value", "details"]].to_string())
    else:
        print(f"\nNo events found for {ticker} in the specified date range")
    
    print(f"\n=== Getting dividend events for {ticker} from {start_date} to {end_date} ===")
    dividend_result = await get_dividends(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh
    )
    
    if "event_data" in dividend_result and not dividend_result["event_data"].empty:
        print(f"\nFound {len(dividend_result['event_data'])} dividend events:")
        print(dividend_result["event_data"][["ticker", "event_date", "event_value", "dividend_type", "yield_value"]].to_string())
    else:
        print(f"\nNo dividend events found for {ticker} in the specified date range")


async def test_event_type(tickers, event_type, start_date, end_date, force_refresh):
    """Test specific event type for multiple tickers"""
    print(f"\n=== Getting {event_type} events for {', '.join(tickers)} from {start_date} to {end_date} ===")
    events_result = await get_corporate_events(
        tickers=tickers,
        event_types=[event_type],
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh
    )
    
    if "event_data" in events_result and not events_result["event_data"].empty:
        print(f"\nFound {len(events_result['event_data'])} {event_type} events:")
        print(events_result["event_data"][["ticker", "event_date", "event_type", "event_value", "details"]].to_string())
    else:
        print(f"\nNo {event_type} events found for the specified tickers in the date range")


if __name__ == "__main__":
    asyncio.run(main()) 