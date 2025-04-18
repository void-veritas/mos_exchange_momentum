# MOEX Index Constituents Tracker

This tool asynchronously fetches and analyzes the constituent stocks of the Moscow Exchange (MOEX) Index over time. It provides historical composition data, FIGI codes, and time series analysis of index changes.

## Features

- **Single Date Retrieval**: Fetch index constituents for any specific date
- **Time Series Analysis**: Track index composition changes over a date range
- **Membership Matrix**: Create a binary (0/1) matrix of stock membership in the index over time
- **Constituent Changes**: Track additions, removals, and weight changes in the index
- **SSL Handling**: Automatically bypass certificate verification issues with MOEX API
- **Data Export**: Save results in CSV and Excel formats for further analysis

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Command Line Interface

```bash
# Fetch current index constituents
python main.py

# Fetch index for a specific date
python main.py 2023-01-15

# Fetch time series data (past 6 months)
python main.py timeseries
```

### Output Files

- `moex_index_membership.csv`: Consolidated matrix of index membership (1=in index, 0=not in index)
- `moex_index_membership.xlsx`: Excel version of the membership matrix
- `moex_index_timeseries.csv`: Detailed data of all stocks across all dates
- `moex_tickers.csv`: Lookup table for ticker symbols and company names
- Individual date snapshots (e.g., `moex_index_2023-01-15.csv`)

## Code Structure

- `main.py`: Command-line interface and example usage
- `moex_api.py`: Core functionality for fetching and processing MOEX data
- `requirements.txt`: Project dependencies

## Advanced Usage

### Fetch Time Series with Different Frequencies

```python
timeseries_data = await fetch_moex_index_timeseries(
    start_date="2022-01-01",
    end_date="2023-01-01",
    frequency="monthly",  # Options: "daily", "weekly", "monthly"
    verify_ssl=False
)
```

### Analyze Index Changes

```python
changes = get_index_changes(timeseries_data)
print(f"Total additions: {changes['summary']['total_additions']}")
print(f"Total removals: {changes['summary']['total_removals']}")
```

## Dependencies

- aiohttp: For async HTTP requests
- pandas: For data processing and export
- certifi: For SSL certificate handling
- openpyxl: For Excel file export 