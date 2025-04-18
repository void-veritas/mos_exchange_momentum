import asyncio
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
from moex_api import fetch_moex_index_stocks, fetch_moex_index_timeseries, get_index_changes


async def fetch_single_date():
    """Fetch MOEX index for a specific date."""
    # Example with a specific date (January 15, 2023)
    specific_date = "2023-01-15"
    print(f"Fetching MOEX index for specific date: {specific_date}")
    stocks = await fetch_moex_index_stocks(date=specific_date, verify_ssl=False)
    
    # Display results
    print(f"\nFound {len(stocks)} stocks in MOEX index for {specific_date}:")
    if stocks:
        print(f"Actual date retrieved: {stocks[0]['date']}")
        
        for i, stock in enumerate(stocks[:5]):  # Show first 5 stocks
            print(f"{i+1}. {stock['ticker']} ({stock['name']}): FIGI = {stock['figi']}, Weight = {stock['weight']}")
        
        if len(stocks) > 5:
            print(f"... and {len(stocks) - 5} more stocks")


async def fetch_timeseries():
    """Fetch MOEX index data as a time series for a date range."""
    # Example with date range (past 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months ago
    
    # Format dates
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Fetching MOEX index time series from {start_str} to {end_str} (monthly frequency)")
    
    # Fetch monthly data for the past 6 months
    timeseries_data = await fetch_moex_index_timeseries(
        start_date=start_str,
        end_date=end_str,
        frequency="monthly",  # Options: "daily", "weekly", "monthly"
        verify_ssl=False
    )
    
    # Display summary
    dates = sorted(timeseries_data.keys())
    print(f"\nRetrieved data for {len(dates)} dates:")
    for date in dates:
        stocks = timeseries_data[date]
        print(f"  {date}: {len(stocks)} stocks")
    
    # Analyze changes in the index over time
    if len(dates) >= 2:
        print("\nAnalyzing changes in index composition:")
        changes = get_index_changes(timeseries_data)
        
        # Display summary of changes
        summary = changes["summary"]
        print(f"- Total periods analyzed: {summary['periods']}")
        print(f"- Total additions: {summary['total_additions']}")
        print(f"- Total removals: {summary['total_removals']}")
        print(f"- Stocks with significant weight changes: {summary['stocks_with_weight_changes']}")
        
        # Display some sample changes if available
        if changes["additions"]:
            first_date = next(iter(changes["additions"]))
            added_stocks = changes["additions"][first_date]
            print(f"\nSample additions on {first_date}:")
            for stock in added_stocks[:3]:  # Show up to 3
                print(f"  + {stock['ticker']} ({stock['name']}) with weight {stock['weight']}%")
        
        if changes["removals"]:
            first_date = next(iter(changes["removals"]))
            removed_stocks = changes["removals"][first_date]
            print(f"\nSample removals on {first_date}:")
            for stock in removed_stocks[:3]:  # Show up to 3
                print(f"  - {stock['ticker']} ({stock['name']})")
    
    # Create binary membership dataframe
    membership_df = create_membership_dataframe(timeseries_data)
    
    # Save the membership dataframe
    if not membership_df.empty:
        membership_file = "moex_index_membership.csv"
        membership_df.to_csv(membership_file)
        print(f"\nSaved consolidated membership dataframe to {membership_file}")
        
        # Also save Excel version for better readability
        excel_file = "moex_index_membership.xlsx"
        membership_df.to_excel(excel_file)
        print(f"Saved Excel version to {excel_file}")
        
        # Display sample of the dataframe
        print("\nSample of the membership dataframe (first 5 rows, 5 columns):")
        sample_cols = list(membership_df.columns[:5])
        print(membership_df[sample_cols].head(5))
    
    # Save data to file
    save_timeseries_to_csv(timeseries_data)


def create_membership_dataframe(timeseries_data):
    """
    Create a consolidated DataFrame with dates as index and companies as columns,
    with 1/0 values indicating index membership.
    
    Args:
        timeseries_data: Dictionary with dates as keys and lists of stock data as values
        
    Returns:
        DataFrame with dates as index, companies as columns, and 1/0 membership values
    """
    if not timeseries_data:
        return pd.DataFrame()
    
    # Get all unique tickers across all dates
    all_tickers = set()
    for stocks in timeseries_data.values():
        all_tickers.update(stock["ticker"] for stock in stocks)
    
    # Sort dates and tickers
    dates = sorted(timeseries_data.keys())
    tickers = sorted(all_tickers)
    
    # Create a DataFrame with dates as index and tickers as columns
    # Initialize with zeros (not in index)
    df = pd.DataFrame(0, index=dates, columns=tickers)
    
    # Fill with 1s where a stock is in the index
    for date, stocks in timeseries_data.items():
        for stock in stocks:
            df.at[date, stock["ticker"]] = 1
    
    # Add additional metadata columns for easier analysis
    df["total_stocks"] = df.sum(axis=1)
    
    # Add ticker names as additional info
    ticker_names = {}
    for stocks in timeseries_data.values():
        for stock in stocks:
            ticker_names[stock["ticker"]] = stock["name"]
    
    # Create a second dataframe with ticker descriptions
    ticker_df = pd.DataFrame({"ticker": tickers, "name": [ticker_names.get(t, "") for t in tickers]})
    ticker_file = "moex_tickers.csv"
    ticker_df.to_csv(ticker_file, index=False)
    print(f"Saved ticker descriptions to {ticker_file}")
    
    return df


def save_timeseries_to_csv(timeseries_data):
    """Save time series data to CSV files for further analysis."""
    if not timeseries_data:
        print("No data to save.")
        return
    
    # Create a snapshot file (each date is a separate CSV)
    for date, stocks in timeseries_data.items():
        if not stocks:
            continue
            
        # Create DataFrame
        df = pd.DataFrame(stocks)
        
        # Save to CSV
        filename = f"moex_index_{date}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved snapshot to {filename}")
    
    # Create a consolidated file (all dates in one file)
    all_data = []
    for date, stocks in timeseries_data.items():
        for stock in stocks:
            all_data.append(stock)
    
    if all_data:
        df_all = pd.DataFrame(all_data)
        consolidated_file = "moex_index_timeseries.csv"
        df_all.to_csv(consolidated_file, index=False)
        print(f"Saved consolidated time series to {consolidated_file}")


async def main():
    """Main function to fetch and display MOEX index constituents."""
    try:
        # Process command line arguments if provided
        if len(sys.argv) > 1:
            # Check if we should run time series mode
            if sys.argv[1] == "timeseries":
                await fetch_timeseries()
            # Check if we need to fetch data for a specific date
            elif len(sys.argv) > 2 and sys.argv[1] == "date":
                specific_date = sys.argv[2]
                print(f"Fetching MOEX index for date: {specific_date}")
                stocks = await fetch_moex_index_stocks(date=specific_date, verify_ssl=False)
                
                # Display results
                print(f"\nFound {len(stocks)} stocks in MOEX index:")
                if stocks:
                    print(f"Date: {stocks[0]['date']}")
                    
                    for i, stock in enumerate(stocks):
                        print(f"{i+1}. {stock['ticker']} ({stock['name']}): FIGI = {stock['figi']}, Weight = {stock['weight']}")
                else:
                    print("No stocks found for the specified date.")
            else:
                # Default to using first argument as a date
                try:
                    datetime.strptime(sys.argv[1], "%Y-%m-%d")
                    specific_date = sys.argv[1]
                    print(f"Fetching MOEX index for date: {specific_date}")
                    stocks = await fetch_moex_index_stocks(date=specific_date, verify_ssl=False)
                    
                    # Display results
                    print(f"\nFound {len(stocks)} stocks in MOEX index:")
                    if stocks:
                        print(f"Date: {stocks[0]['date']}")
                        
                        for i, stock in enumerate(stocks):
                            print(f"{i+1}. {stock['ticker']} ({stock['name']}): FIGI = {stock['figi']}, Weight = {stock['weight']}")
                    else:
                        print("No stocks found for the specified date.")
                except ValueError:
                    print(f"Invalid date format: {sys.argv[1]}. Use YYYY-MM-DD format.")
                    print("Usage examples:")
                    print("  python main.py                  # Fetch current date")
                    print("  python main.py 2023-01-15       # Fetch specific date")
                    print("  python main.py date 2023-01-15  # Same as above")
                    print("  python main.py timeseries       # Fetch time series (past 6 months)")
        else:
            # No arguments - run in demo mode with both examples
            print("Running in demo mode...")
            await fetch_single_date()
            print("\n" + "-"*80 + "\n")
            await fetch_timeseries()
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
