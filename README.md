# Moscow Exchange Data API

A Python application for retrieving stock data from the Moscow Exchange and various other sources.

## Project Overview

This project provides tools for fetching, storing, and analyzing stock data from multiple sources including:

- Moscow Exchange (MOEX)
- Yahoo Finance
- Tinkoff API
- Financial Modeling Prep (FMP)
- Finam
- Investing.com

## Features

- Fetch current MOEX index constituents
- Track changes in index composition over time
- Retrieve historical price data from multiple sources
- Store data in MongoDB or CSV files
- Analyze price movements and index changes

## Project Structure

The project follows a clean architecture approach with the following structure:

```
mos_exchange/
├── src/                           # Source code
│   ├── domain/                    # Domain layer (business logic)
│   │   ├── entities/              # Business entities
│   │   │   ├── __init__.py
│   │   │   ├── price.py           # Price data entity
│   │   │   ├── security.py        # Security/stock entity
│   │   │   └── index.py           # Index composition entity
│   │   ├── interfaces/            # Abstract repositories & services
│   │   │   ├── __init__.py
│   │   │   ├── price_repository.py
│   │   │   └── data_source.py
│   │   └── services/              # Business use cases
│   │       ├── __init__.py
│   │       ├── price_service.py
│   │       └── index_service.py
│   ├── infrastructure/            # External interfaces implementation
│   │   ├── __init__.py
│   │   ├── data_sources/          # Data source implementations
│   │   │   ├── __init__.py
│   │   │   ├── moex_api.py
│   │   │   ├── yahoo_api.py
│   │   │   ├── tinkoff_api.py
│   │   │   └── finam_api.py
│   │   ├── repositories/          # Repository implementations
│   │   │   ├── __init__.py
│   │   │   ├── mongodb_repository.py
│   │   │   ├── sqlite_repository.py
│   │   │   └── csv_repository.py
│   │   └── utils/                 # Utility functions
│   │       ├── __init__.py
│   │       ├── date_utils.py
│   │       └── error_handling.py
│   └── application/               # Application layer
│       ├── __init__.py
│       ├── dto/                   # Data Transfer Objects
│       │   ├── __init__.py
│       │   └── price_dto.py
│       ├── services/              # Application services
│       │   ├── __init__.py
│       │   ├── price_fetcher.py
│       │   └── price_storage.py
│       └── config.py              # Application configuration
├── scripts/                       # Command-line scripts
│   ├── fetch_prices.py
│   └── fetch_index.py
├── tests/                         # Unit tests
│   ├── domain/
│   ├── infrastructure/
│   └── application/
├── .gitignore
├── requirements.txt
└── README.md
```

## Clean Architecture Benefits

This structure provides several benefits:

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Dependency Rule**: Dependencies flow inward, domain layer has no external dependencies
3. **Testability**: Business logic is isolated and easily testable
4. **Flexibility**: Data sources and storage mechanisms can be changed without affecting business logic

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

Fetch current index composition:

```bash
python scripts/fetch_index.py
```

Fetch price data for specific tickers:

```bash
python scripts/fetch_prices.py --tickers SBER,GAZP,LKOH --source yahoo --start-date 2023-01-01
```

## Data Sources

The application can fetch data from:

- **MOEX API**: Moscow Exchange official API for index compositions and price data
- **Yahoo Finance**: Global price data source with good historical coverage
- **Tinkoff API**: Russian broker API requiring authentication
- **Finam**: Russian financial data provider
- **Financial Modeling Prep**: Alternative global financial data API
- **Investing.com**: Web-based financial data source

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
This project is licensed under the MIT License - see the LICENSE file for details. 