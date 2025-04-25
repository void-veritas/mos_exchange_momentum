#!/usr/bin/env python3
import asyncio
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.entities.corporate_event import EventType
from src.domain.entities.price import Price
from src.application.services.corporate_event_fetcher import CorporateEventFetcherService
from src.application.services.price_fetcher import PriceFetcherService
from src.domain.services.corporate_event_service import CorporateEventService
from src.infrastructure.repositories.corporate_event_repository import CorporateEventRepository
from src.infrastructure.repositories.price_repository import PriceRepository
from src.infrastructure.data_sources.moex_api import MOEXDataSource
from src.infrastructure.data_sources.fmp_api import FMPDataSource
from src.domain.services.price_service import PriceService


async def fetch_prices_with_adjustments(ticker="SBER", start_date="2000-01-01"):
    """
    Fetch complete price history with corporate events and demonstrate
    different price adjustment methods
    
    Args:
        ticker: Ticker symbol to analyze
        start_date: Start date for historical data
    """
    print(f"\n{'='*80}\nAnalyzing complete price history for {ticker} with corporate events\n{'='*80}")
    
    # Create repositories and data sources
    event_repository = CorporateEventRepository()
    price_repository = PriceRepository()
    
    data_sources = [
        MOEXDataSource(),
        FMPDataSource()
    ]
    
    # Create services
    event_service = CorporateEventService(event_repository, data_sources)
    event_fetcher = CorporateEventFetcherService(event_service)
    
    # Create price service first, then pass it to the price fetcher
    price_service = PriceService(price_repository, data_sources)
    price_fetcher = PriceFetcherService(price_service)
    
    # Get end date as today
    end_date = date.today().isoformat()
    
    print(f"\nFetching price data for {ticker} from {start_date} to {end_date}...")
    
    # Fetch historical prices
    prices_result = await price_fetcher.fetch_prices(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=True,
        as_dataframe=True
    )
    
    if "price_data" not in prices_result or prices_result["price_data"].empty:
        print(f"No price data found for {ticker}. Using sample data for demonstration.")
        # Create sample price data for demonstration
        prices_df = create_sample_price_data(ticker, start_date, end_date)
    else:
        prices_df = prices_result["price_data"]
    
    print(f"\nFetched {len(prices_df)} price records for {ticker}.")
    print("\nSample of original price data:")
    print(prices_df.head().to_string())
    
    # Fetch all corporate events
    print(f"\nFetching corporate events for {ticker} from {start_date} to {end_date}...")
    events_result = await event_fetcher.get_events(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        force_refresh=True,
        as_dataframe=True
    )
    
    if "event_data" not in events_result or events_result["event_data"].empty:
        print(f"No corporate events found for {ticker}. Using sample events for demonstration.")
        # Create sample events for demonstration
        events_df = create_sample_events(ticker, start_date, end_date)
    else:
        events_df = events_result["event_data"]
    
    print(f"\nFetched {len(events_df)} corporate events for {ticker}.")
    print("\nCorporate events summary:")
    if not events_df.empty:
        event_summary = events_df.groupby('event_type').size().reset_index(name='count')
        print(event_summary.to_string())
        print("\nSample of corporate events:")
        print(events_df[['event_date', 'event_type', 'event_value', 'details']].head().to_string())
    
    # Create a merged dataframe with prices and events
    # First convert events_df to events per date format
    if not events_df.empty:
        events_by_date = events_df[['event_date', 'event_type', 'event_value', 'details']].copy()
        events_by_date['event_date'] = pd.to_datetime(events_by_date['event_date'])
        events_by_date.set_index('event_date', inplace=True)
    else:
        events_by_date = pd.DataFrame()
    
    # Set index on prices dataframe for merging
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_with_events = prices_df.set_index('date').copy()
    
    # Merge with events
    if not events_by_date.empty:
        # Add event type flags to prices dataframe
        for event_type in events_by_date['event_type'].unique():
            event_filter = events_by_date['event_type'] == event_type
            event_dates = events_by_date[event_filter].index
            prices_with_events[f'has_{event_type}'] = False
            
            # Only set flags for dates that exist in the price data
            valid_dates = [date for date in event_dates if date in prices_with_events.index]
            if valid_dates:
                prices_with_events.loc[valid_dates, f'has_{event_type}'] = True
            else:
                print(f"Warning: No matching trading days found for {event_type} events")
            
            # Add event values where applicable
            prices_with_events[f'{event_type}_value'] = np.nan
            for date_idx in event_dates:
                if date_idx in prices_with_events.index:
                    event_value = events_by_date.loc[date_idx, 'event_value']
                    if isinstance(event_value, pd.Series):
                        event_value = event_value.iloc[0]
                    prices_with_events.loc[date_idx, f'{event_type}_value'] = event_value
                else:
                    # For logging/debugging: Event date doesn't have a corresponding trading day
                    print(f"Note: Event date {date_idx.date()} ({event_type}) is not a trading day in price data")
    
    # Reset index for easier handling
    prices_with_events = prices_with_events.reset_index()
    print("\nMerged prices with events data sample:")
    print(prices_with_events.head().to_string())
    
    # Create adjusted price columns
    
    # 1. Backward adjustment (adjust all prices before the event)
    # - For dividends: add dividend amount to prices before ex-date (additive)
    # - For splits: multiply prices before split date by split ratio (multiplicative)
    
    # Clone original price data for adjustments
    backward_adj_df = prices_with_events.copy()
    backward_adj_df['close_adj_backward'] = backward_adj_df['close']
    backward_adj_df['dividend_adjustment'] = 0.0
    backward_adj_df['split_adjustment'] = 1.0
    
    # 2. Forward adjustment (adjust all prices after the event)
    # - For dividends: subtract dividend amount from prices after ex-date (additive)
    # - For splits: divide prices after split date by split ratio (multiplicative)
    
    forward_adj_df = prices_with_events.copy()
    forward_adj_df['close_adj_forward'] = forward_adj_df['close'] 
    forward_adj_df['dividend_adjustment'] = 0.0
    forward_adj_df['split_adjustment'] = 1.0
    
    # Apply dividend adjustments if we have dividend events
    if 'has_dividend' in prices_with_events.columns and prices_with_events['has_dividend'].any():
        print("\nApplying dividend adjustments...")
        
        # Get all dividend events and sort by date (oldest first for backward adjustment)
        dividend_events = events_df[events_df['event_type'] == 'dividend'].sort_values('event_date')
        
        # Backward adjustment - we start from the oldest and work forward
        cum_div_adj = 0
        for idx, event in dividend_events.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            event_value = event['event_value']
            
            if pd.isna(event_value):
                continue
                
            # Backward adjustment (add dividend to all prices before the ex-date)
            cum_div_adj += event_value
            backward_adj_df.loc[backward_adj_df['date'] < event_date, 'dividend_adjustment'] += event_value
        
        # Apply the cumulative adjustment
        backward_adj_df['close_adj_backward'] = backward_adj_df['close'] + backward_adj_df['dividend_adjustment']
        
        # Forward adjustment - we start from today and work backward
        # Reverse the event order for forward adjustment
        dividend_events = dividend_events.sort_values('event_date', ascending=False)
        cum_div_adj = 0
        for idx, event in dividend_events.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            event_value = event['event_value']
            
            if pd.isna(event_value):
                continue
                
            # Forward adjustment (subtract dividend from all prices after the ex-date)
            cum_div_adj += event_value
            forward_adj_df.loc[forward_adj_df['date'] >= event_date, 'dividend_adjustment'] += event_value
        
        # Apply the cumulative adjustment
        forward_adj_df['close_adj_forward'] = forward_adj_df['close'] - forward_adj_df['dividend_adjustment']
    
    # Apply split adjustments if we have split events
    if 'has_stock_split' in prices_with_events.columns and prices_with_events['has_stock_split'].any():
        print("\nApplying stock split adjustments...")
        
        # Get all split events and sort by date (oldest first for backward adjustment)
        split_events = events_df[events_df['event_type'] == 'stock_split'].sort_values('event_date')
        
        # Backward adjustment - we start from the oldest and work forward
        cum_split_factor = 1.0
        for idx, event in split_events.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            split_ratio = event['event_value']
            
            if pd.isna(split_ratio):
                continue
                
            # Backward adjustment (multiply all prices before the split date by the split ratio)
            cum_split_factor *= split_ratio
            backward_adj_df.loc[backward_adj_df['date'] < event_date, 'split_adjustment'] *= split_ratio
        
        # Apply the cumulative adjustment
        backward_adj_df['close_adj_backward'] = backward_adj_df['close_adj_backward'] * backward_adj_df['split_adjustment']
        
        # Forward adjustment - we start from today and work backward
        # Reverse the event order for forward adjustment
        split_events = split_events.sort_values('event_date', ascending=False)
        cum_split_factor = 1.0
        for idx, event in split_events.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            split_ratio = event['event_value']
            
            if pd.isna(split_ratio):
                continue
                
            # Forward adjustment (divide all prices after the split date by the split ratio)
            cum_split_factor *= split_ratio
            forward_adj_df.loc[forward_adj_df['date'] >= event_date, 'split_adjustment'] *= split_ratio
        
        # Apply the cumulative adjustment
        forward_adj_df['close_adj_forward'] = forward_adj_df['close_adj_forward'] / forward_adj_df['split_adjustment']
    
    # Display results
    print("\nBackward adjustment results (sample):")
    print(backward_adj_df[['date', 'close', 'dividend_adjustment', 'split_adjustment', 'close_adj_backward']].head().to_string())
    
    print("\nForward adjustment results (sample):")
    print(forward_adj_df[['date', 'close', 'dividend_adjustment', 'split_adjustment', 'close_adj_forward']].head().to_string())
    
    # Save the data to CSV files
    backward_adj_df.to_csv(f'data/{ticker}_backward_adjusted.csv', index=False)
    forward_adj_df.to_csv(f'data/{ticker}_forward_adjusted.csv', index=False)
    events_df.to_csv(f'data/{ticker}_corporate_events.csv', index=False)
    
    print(f"\nSaved adjusted price data to data/{ticker}_backward_adjusted.csv and data/{ticker}_forward_adjusted.csv")
    print(f"Saved corporate events data to data/{ticker}_corporate_events.csv")
    
    # Export to Excel
    export_to_excel(ticker, prices_df, events_df, backward_adj_df, forward_adj_df)
    
    # Plot the results
    plot_price_adjustments(ticker, prices_with_events, backward_adj_df, forward_adj_df, events_df)


def create_sample_price_data(ticker, start_date, end_date):
    """
    Create sample price data for demonstration purposes
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with sample price data
    """
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range with business days
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    # Start with base price and add random walk
    base_price = 100
    np.random.seed(42)  # For reproducibility
    
    # Generate random walk with positive drift
    returns = np.random.normal(0.0003, 0.02, size=len(date_range))
    # Add occasional jumps
    jumps = np.random.normal(0, 0.1, size=len(date_range))
    jumps[jumps < 0.2] = 0
    
    returns += jumps
    
    # Convert to price series
    log_returns = np.cumsum(returns)
    prices = base_price * np.exp(log_returns)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': date_range,
        'ticker': ticker,
        'open': prices * (1 + np.random.normal(0, 0.005, size=len(date_range))),
        'high': prices * (1 + np.random.normal(0.01, 0.01, size=len(date_range))),
        'low': prices * (1 - np.random.normal(0.01, 0.01, size=len(date_range))),
        'close': prices,
        'volume': np.random.normal(1000000, 500000, size=len(date_range))
    })
    
    # Ensure high > low
    df['high'] = df[['high', 'low']].max(axis=1) * 1.01
    
    return df


def create_sample_events(ticker, start_date, end_date):
    """
    Create sample corporate events for demonstration purposes
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with sample events
    """
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create some sample events
    events = []
    
    # Add dividends (assume quarterly dividends)
    current_date = start + pd.DateOffset(months=3)
    dividend_value = 0.5
    
    while current_date < end:
        # Gradually increase dividend over time
        dividend_value *= 1.02
        
        events.append({
            'ticker': ticker,
            'event_date': current_date,
            'event_type': 'dividend',
            'event_value': round(dividend_value, 2),
            'details': f'Quarterly dividend payment',
            'dividend_type': 'quarterly',
            'currency': 'RUB',
            'yield_value': round(dividend_value / 100 * 100, 2),  # Assume 100 is price at that time
        })
        
        current_date += pd.DateOffset(months=3)
    
    # Add some stock splits
    splits = [
        {'date': start + pd.DateOffset(years=5), 'ratio': 2.0},
        {'date': start + pd.DateOffset(years=10), 'ratio': 2.0},
        {'date': start + pd.DateOffset(years=15), 'ratio': 1.5}
    ]
    
    for split in splits:
        if split['date'] < end:
            events.append({
                'ticker': ticker,
                'event_date': split['date'],
                'event_type': 'stock_split',
                'event_value': split['ratio'],
                'details': f'{split["ratio"]}-for-1 stock split',
            })
    
    # Add a name change
    name_change_date = start + pd.DateOffset(years=7)
    if name_change_date < end:
        events.append({
            'ticker': ticker,
            'event_date': name_change_date,
            'event_type': 'name_change',
            'event_value': None,
            'details': f'Changed name from "Old {ticker}" to "{ticker}"',
        })
    
    # Create DataFrame
    events_df = pd.DataFrame(events)
    
    # Sort by date
    events_df = events_df.sort_values('event_date')
    
    return events_df


def plot_price_adjustments(ticker, original, backward_adj, forward_adj, events_df):
    """
    Plot original and adjusted price series with event markers
    
    Args:
        ticker: Ticker symbol
        original: DataFrame with original prices
        backward_adj: DataFrame with backward adjusted prices
        forward_adj: DataFrame with forward adjusted prices
        events_df: DataFrame with corporate events
    """
    plt.figure(figsize=(15, 10))
    
    # Plot price series
    plt.subplot(2, 1, 1)
    plt.plot(original['date'], original['close'], label='Original Close', color='blue', alpha=0.5)
    plt.plot(backward_adj['date'], backward_adj['close_adj_backward'], 
             label='Backward Adjusted', color='green', linewidth=1.5)
    plt.plot(forward_adj['date'], forward_adj['close_adj_forward'], 
             label='Forward Adjusted', color='red', linewidth=1.5)
    
    # Mark events on the chart
    if not events_df.empty:
        for event_type in events_df['event_type'].unique():
            type_events = events_df[events_df['event_type'] == event_type]
            for idx, event in type_events.iterrows():
                event_date = pd.to_datetime(event['event_date'])
                event_value = event['event_value']
                
                if event_type == 'dividend':
                    marker = 'o'
                    color = 'purple'
                    label = f'Dividend: {event_value:.2f}' if not pd.isna(event_value) else 'Dividend'
                elif event_type == 'stock_split':
                    marker = 's'
                    color = 'orange'
                    label = f'Split: {event_value:.2f}:1' if not pd.isna(event_value) else 'Split'
                else:
                    marker = '^'
                    color = 'black'
                    label = f'{event_type}'
                
                # Find closest price point
                closest_idx = (original['date'] - event_date).abs().idxmin()
                price_at_event = original.iloc[closest_idx]['close']
                
                plt.scatter([event_date], [price_at_event], marker=marker, color=color, 
                           s=100, label=label)
    
    plt.title(f'{ticker} Price History with Adjustments')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Use a logarithmic y-scale to better visualize long-term price changes
    plt.yscale('log')
    
    # Create custom legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    # Plot returns comparison
    plt.subplot(2, 1, 2)
    
    # Calculate returns
    original['returns'] = original['close'].pct_change()
    backward_adj['returns'] = backward_adj['close_adj_backward'].pct_change()
    forward_adj['returns'] = forward_adj['close_adj_forward'].pct_change()
    
    # Calculate cumulative returns
    original['cum_returns'] = (1 + original['returns']).cumprod() - 1
    backward_adj['cum_returns'] = (1 + backward_adj['returns']).cumprod() - 1
    forward_adj['cum_returns'] = (1 + forward_adj['returns']).cumprod() - 1
    
    plt.plot(original['date'], original['cum_returns'] * 100, 
             label='Original Returns', color='blue', alpha=0.5)
    plt.plot(backward_adj['date'], backward_adj['cum_returns'] * 100, 
             label='Backward Adjusted Returns', color='green', linewidth=1.5)
    plt.plot(forward_adj['date'], forward_adj['cum_returns'] * 100, 
             label='Forward Adjusted Returns', color='red', linewidth=1.5)
    
    plt.title(f'{ticker} Cumulative Returns Comparison')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'data/{ticker}_price_adjustments.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved price adjustment chart to data/{ticker}_price_adjustments.png")


def export_to_excel(ticker, prices_df, events_df, backward_adj_df, forward_adj_df):
    """
    Export all data to a single Excel file with multiple sheets
    
    Args:
        ticker: Ticker symbol
        prices_df: Original prices dataframe
        events_df: Corporate events dataframe
        backward_adj_df: Backward adjusted prices dataframe 
        forward_adj_df: Forward adjusted prices dataframe
    """
    excel_file = f'data/{ticker}_price_analysis.xlsx'
    
    # Create a Pandas Excel writer
    writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    
    # Write each dataframe to a different worksheet
    prices_df.to_excel(writer, sheet_name='Original Prices', index=False)
    events_df.to_excel(writer, sheet_name='Corporate Events', index=False)
    
    # Create a selection of columns for the adjusted prices
    backward_columns = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 
        'volume', 'dividend_adjustment', 'split_adjustment', 'close_adj_backward'
    ]
    forward_columns = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 
        'volume', 'dividend_adjustment', 'split_adjustment', 'close_adj_forward'
    ]
    
    # Filter columns that actually exist in the dataframes
    backward_columns = [col for col in backward_columns if col in backward_adj_df.columns]
    forward_columns = [col for col in forward_columns if col in forward_adj_df.columns]
    
    # Write the adjusted dataframes
    backward_adj_df[backward_columns].to_excel(writer, sheet_name='Backward Adjusted', index=False)
    forward_adj_df[forward_columns].to_excel(writer, sheet_name='Forward Adjusted', index=False)
    
    # Create a summary sheet
    summary_data = {
        'Category': ['Original Prices', 'Corporate Events', 'Backward Adjusted', 'Forward Adjusted'],
        'Row Count': [len(prices_df), len(events_df), len(backward_adj_df), len(forward_adj_df)],
        'Description': [
            'Original unadjusted price data',
            'Corporate events affecting price adjustments',
            'Prices adjusted backward in time',
            'Prices adjusted forward in time'
        ]
    }
    
    if not events_df.empty:
        # Add event type summary
        event_summary = events_df.groupby('event_type').size().reset_index(name='count')
        event_summary_data = {
            'Event Type': event_summary['event_type'].tolist(),
            'Count': event_summary['count'].tolist()
        }
        event_summary_df = pd.DataFrame(event_summary_data)
        event_summary_df.to_excel(writer, sheet_name='Event Summary', index=False)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Save the Excel file
    writer.close()
    
    print(f"\nExported all data to Excel file: {excel_file}")


if __name__ == "__main__":
    # Set the ticker from command line if provided, otherwise use SBER
    ticker = "SBER"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    # Run the main function
    asyncio.run(fetch_prices_with_adjustments(ticker)) 