#!/usr/bin/env python3
import asyncio
import argparse
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application components
from src.application.config import Config
from src.application.services.price_fetcher import PriceFetcherService
from src.domain.services.price_service import PriceService
from src.infrastructure.repositories.sqlite_repository import SQLitePriceRepository

# Import data sources
from src.infrastructure.data_sources.yahoo_api import YahooFinanceDataSource
from src.infrastructure.data_sources.fmp_api import FMPDataSource


async def create_services():
    """Create application services"""
    # Initialize repositories
    repository = SQLitePriceRepository(db_path=Config.DB_PATH)
    
    # Initialize data sources
    data_sources = []
    
    # Create FMP data source if API key is configured
    if Config.FMP_API_KEY:
        fmp_source = FMPDataSource()
        data_sources.append(fmp_source)
        logger.info("FMP data source initialized")
    
    # Create Yahoo Finance data source
    yahoo_source = YahooFinanceDataSource()
    data_sources.append(yahoo_source)
    logger.info("Yahoo Finance data source initialized")
    
    # Create domain services
    price_service = PriceService(repository=repository, data_sources=data_sources)
    
    # Create application services
    price_fetcher = PriceFetcherService(price_service=price_service)
    
    return price_fetcher


async def main():
    """Main function to fetch price data"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch stock price data from various sources")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of ticker symbols")
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--force-refresh", action="store_true", help="Force fetching new data (ignore cache)")
    parser.add_argument("--format", type=str, choices=["json", "dataframe"], default="json", 
                        help="Output format (json or dataframe)")
    parser.add_argument("--output-file", type=str, help="Output file (if not specified, prints to console)")
    
    args = parser.parse_args()
    
    # Parse ticker list
    tickers = [t.strip() for t in args.tickers.split(",")]
    
    try:
        # Initialize configuration
        Config.init()
        
        # Create services
        price_fetcher = await create_services()
        
        # Fetch prices
        logger.info(f"Fetching prices for {len(tickers)} tickers")
        result = await price_fetcher.fetch_prices(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
            as_dataframe=(args.format == "dataframe")
        )
        
        # Process output
        if args.format == "json":
            # For JSON output, convert DTOs to dictionaries
            output = {}
            for ticker, data in result.items():
                if "prices" in data:
                    # Convert price DTOs to dictionaries
                    prices_dicts = [p.to_dict() for p in data["prices"]]
                    output[ticker] = {
                        "ticker": ticker,
                        "prices": prices_dicts,
                        "start_date": str(data["start_date"]) if data["start_date"] else None,
                        "end_date": str(data["end_date"]) if data["end_date"] else None,
                        "data_points": data["data_points"]
                    }
                else:
                    # For dataframe output, convert DataFrames to dictionaries
                    df = data["price_data"]
                    output[ticker] = {
                        "ticker": ticker,
                        "prices": json.loads(df.to_json(orient="records")),
                        "start_date": str(data["start_date"]) if data["start_date"] else None,
                        "end_date": str(data["end_date"]) if data["end_date"] else None,
                        "data_points": data["data_points"]
                    }
            
            # Convert to JSON string
            json_data = json.dumps(output, indent=2)
            
            # Output
            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(json_data)
                logger.info(f"Results written to {args.output_file}")
            else:
                print(json_data)
        else:
            # For dataframe output, print summary
            print("\nResults summary:")
            print("=" * 50)
            for ticker, data in result.items():
                df = data["price_data"]
                print(f"{ticker}: {len(df)} records from {data['start_date']} to {data['end_date']}")
                if not df.empty:
                    print(f"\nSample data for {ticker}:")
                    print(df.head(3))
            print("=" * 50)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 