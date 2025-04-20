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
from src.domain.services.index_service import IndexService
from src.infrastructure.data_sources.moex_api import MoexIndexDataSource


async def create_services():
    """Create application services"""
    # Initialize data sources
    moex_source = MoexIndexDataSource()
    
    # Create domain services
    index_service = IndexService(data_sources=[moex_source])
    
    return index_service


async def main():
    """Main function to fetch index data"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch stock index data from MOEX")
    parser.add_argument("--index", type=str, default="IMOEX", help="Index ID (default: IMOEX)")
    parser.add_argument("--date", type=str, help="Specific date in YYYY-MM-DD format (default: latest)")
    parser.add_argument("--start-date", type=str, help="Start date for time series in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date for time series in YYYY-MM-DD format")
    parser.add_argument("--frequency", type=str, choices=["daily", "weekly", "monthly"], default="monthly",
                        help="Frequency for time series (daily, weekly, monthly)")
    parser.add_argument("--timeseries", action="store_true", help="Fetch time series instead of single date")
    parser.add_argument("--analyze", action="store_true", help="Analyze changes in index composition")
    parser.add_argument("--output-file", type=str, help="Output file (if not specified, prints to console)")
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        Config.init()
        
        # Create services
        index_service = await create_services()
        
        # Fetch data
        if args.timeseries:
            logger.info(f"Fetching {args.index} index time series from {args.start_date or 'default'} to {args.end_date or 'today'} ({args.frequency})")
            result = await index_service.get_index_timeseries(
                index_id=args.index,
                start_date=args.start_date,
                end_date=args.end_date,
                frequency=args.frequency
            )
            
            # Convert to dictionary for JSON serialization
            output = {}
            for date_str, index in result.items():
                output[date_str] = index.to_dict()
            
            # Analyze if requested
            if args.analyze and result:
                analysis = index_service.analyze_index_changes(result)
                output["analysis"] = analysis
        
        else:
            logger.info(f"Fetching {args.index} index composition for {args.date or 'latest'}")
            result = await index_service.get_index_composition(
                index_id=args.index,
                date_str=args.date
            )
            
            # Convert to dictionary for JSON serialization
            if result:
                output = result.to_dict()
            else:
                output = {"error": f"No data found for {args.index} on {args.date or 'latest'}"}
        
        # Output
        json_data = json.dumps(output, indent=2)
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(json_data)
            logger.info(f"Results written to {args.output_file}")
        else:
            print(json_data)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 