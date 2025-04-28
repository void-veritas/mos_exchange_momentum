#!/usr/bin/env python3
"""
Умная Бета-Стратегия для акций из индекса Московской биржи с максимизацией Sharpe Ratio.
Включает ежемесячную ребалансировку и учет комиссий при бэктестировании.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any

# Добавление директории src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.entities.portfolio import Portfolio
from src.domain.services.price_service import PriceService
from src.infrastructure.repositories.price_repository import PriceRepository
from src.infrastructure.data_sources.moex_api import MOEXDataSource, MoexIndexDataSource
from src.infrastructure.data_sources.yahoo_api import YahooFinanceDataSource

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
COMMISSION_RATE = 0.001  # 0.1% комиссия на сделку
REBALANCE_FREQUENCY = 21  # ~21 торговый день в месяце
LOOKBACK_PERIOD = 252  # Годовой период для расчета метрик

async def fetch_moex_index_stocks() -> List[str]:
    """
    Получение списка акций из индекса Московской биржи
    
    Returns:
        Список тикеров акций
    """
    logger.info("Получение состава индекса Московской биржи")
    
    index_source = MoexIndexDataSource()
    try:
        # Получение текущего состава индекса
        moex_index = await index_source.fetch_index_composition(index_id="IMOEX")
        if moex_index and moex_index.constituents:
            tickers = [c.security.ticker for c in moex_index.constituents]
            logger.info(f"Получено {len(tickers)} акций из индекса IMOEX")
            return tickers
        else:
            logger.warning("Не удалось получить данные индекса IMOEX, используем тестовый набор")
            # Тестовый набор акций индекса MOEX
            return ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS", "YNDX", "MGNT", 
                    "ALRS", "CHMF", "MOEX", "NLMK", "VTBR", "POLY", "SNGS", "IRAO", "FIVE", "AFLT"]
    except Exception as e:
        logger.error(f"Ошибка при получении состава индекса: {e}")
        # Тестовый набор акций индекса MOEX в случае ошибки
        return ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MTSS", "YNDX", "MGNT", 
                "ALRS", "CHMF", "MOEX", "NLMK", "VTBR", "POLY", "SNGS", "IRAO", "FIVE", "AFLT"]

async def load_price_data(tickers: List[str], start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Загрузка ценовых данных для указанных тикеров
    
    Args:
        tickers: Список тикеров для загрузки
        start_date: Начальная дата в формате YYYY-MM-DD
        end_date: Конечная дата в формате YYYY-MM-DD
        
    Returns:
        DataFrame с ценами закрытия для всех тикеров
    """
    logger.info(f"Загрузка ценовых данных для {len(tickers)} тикеров")
    
    # Создание сервисов
    price_repo = PriceRepository()
    data_sources = [YahooFinanceDataSource(), MOEXDataSource()]
    price_service = PriceService(repository=price_repo, data_sources=data_sources)
    
    try:
        # Загрузка цен
        prices_dict = await price_service.get_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        # Преобразование в DataFrame
        prices_df = pd.DataFrame()
        for ticker, price_list in prices_dict.items():
            if price_list:
                # Извлечение дат и цен закрытия
                dates = [p.date for p in price_list]
                close_prices = [p.close or p.adjusted_close for p in price_list]
                ticker_df = pd.DataFrame({ticker: close_prices}, index=dates)
                prices_df = pd.concat([prices_df, ticker_df], axis=1)
        
        if prices_df.empty or len(prices_df) < 30:
            logger.warning("Недостаточно данных, создаем синтетические данные")
            prices_df = create_sample_price_data(tickers)
        else:
            logger.info(f"Успешно загружены данные для {len(prices_df.columns)} тикеров, {len(prices_df)} дней")
        
        return prices_df
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        logger.info("Создание синтетических данных")
        return create_sample_price_data(tickers)

def create_sample_price_data(tickers: List[str]) -> pd.DataFrame:
    """
    Создание синтетических ценовых данных для тестирования
    
    Args:
        tickers: Список тикеров
        
    Returns:
        DataFrame с ценами закрытия
    """
    # Создание временного ряда на 2 года
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Генерация бизнес-дней
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Инициализация DataFrame
    prices = pd.DataFrame(index=date_range)
    
    # Генерация случайных цен для каждого тикера
    np.random.seed(42)  # Для воспроизводимости
    
    for ticker in tickers:
        # Начальная цена от 100 до 1000
        initial_price = np.random.uniform(100, 1000)
        
        # Дневная доходность с дрифтом
        drift = np.random.uniform(0.0001, 0.0005)  # Небольшой положительный дрифт
        volatility = np.random.uniform(0.01, 0.02)  # Дневная волатильность
        
        # Генерация логарифмических доходностей
        log_returns = np.random.normal(drift, volatility, len(date_range))
        
        # Преобразование в цены
        price_series = initial_price * np.exp(np.cumsum(log_returns))
        
        # Добавление в DataFrame
        prices[ticker] = price_series
    
    logger.info(f"Создан синтетический набор данных: {len(prices)} дней, {len(tickers)} тикеров")
    return prices

def create_factor_data(tickers: List[str], prices: pd.DataFrame) -> pd.DataFrame:
    """
    Создание факторной модели на основе ценовых данных
    
    Args:
        tickers: Список тикеров
        prices: DataFrame с ценами
        
    Returns:
        DataFrame с факторами для каждого тикера
    """
    # Создание случайных отраслей и секторов
    industries = ["Финансы", "Нефть и газ", "Телеком", "Металлургия", "Ритейл", 
                  "Технологии", "Химия", "Энергетика", "Транспорт", "Строительство"]
    sectors = ["Финансовый", "Сырьевой", "Коммуникационный", "Промышленный", "Потребительский"]
    
    # Создание DataFrame с факторами
    factors = pd.DataFrame(index=tickers)
    
    # Назначение случайных отраслей и секторов
    np.random.seed(42)  # Для воспроизводимости
    
    factors["industry"] = np.random.choice(industries, size=len(tickers))
    factors["sector"] = np.random.choice(sectors, size=len(tickers))
    
    # Расчет факторов на основе исторических данных
    if not prices.empty:
        returns = prices.pct_change().dropna()
        
        # Расчет среднедневной доходности
        factors["factor_daily_return"] = returns.mean()
        
        # Расчет волатильности (добавляем префикс, чтобы избежать конфликта имен)
        factors["factor_volatility"] = returns.std()
        
        # Расчет моментума (доходность за последние 3 месяца) с префиксом
        if len(prices) >= 63:  # ~3 месяца торговых дней
            start_prices = prices.iloc[-63]
            end_prices = prices.iloc[-1]
            factors["factor_momentum"] = (end_prices / start_prices) - 1
        else:
            # Если недостаточно данных, используем имеющиеся
            start_prices = prices.iloc[0]
            end_prices = prices.iloc[-1]
            factors["factor_momentum"] = (end_prices / start_prices) - 1
        
        # Расчет бета коэффициента к индексу (используем среднее значение как аппроксимацию индекса)
        market_returns = returns.mean(axis=1)
        for ticker in returns.columns:
            cov = returns[ticker].cov(market_returns)
            market_var = market_returns.var()
            if market_var > 0:
                factors.loc[ticker, "factor_beta"] = cov / market_var
            else:
                factors.loc[ticker, "factor_beta"] = 1.0
    
    return factors

async def run_smart_beta_strategy(start_date: Optional[str] = None, 
                                  end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Запуск бэктеста умной бета-стратегии
    
    Args:
        start_date: Начальная дата в формате YYYY-MM-DD
        end_date: Конечная дата в формате YYYY-MM-DD
        
    Returns:
        Словарь с результатами бэктеста
    """
    logger.info("Запуск умной бета-стратегии с максимизацией Sharpe Ratio")
    
    # Получение акций из индекса MOEX
    tickers = await fetch_moex_index_stocks()
    
    # Загрузка ценовых данных
    prices = await load_price_data(tickers, start_date, end_date)
    
    # Создание факторов для акций
    factors = create_factor_data(tickers, prices)
    
    # Создание портфеля с оптимизацией по Sharpe Ratio
    portfolio = Portfolio(factors, select_by="optimize", select_metric="sharpe")
    
    # Запуск бэктеста стратегии
    results = run_backtest(prices, portfolio)
    
    # Расчет доходности buy-and-hold для сравнения
    buyhold_returns = calculate_buyhold_returns(prices)
    
    # Построение графиков
    plot_results(results, buyhold_returns)
    
    # Расчет метрик эффективности
    performance = calculate_performance_metrics(results, buyhold_returns)
    
    return {
        'prices': prices,
        'factors': factors,
        'strategy_returns': results['strategy_returns'],
        'cumulative_returns': results['cumulative_returns'],
        'portfolio_weights': results['portfolio_weights'],
        'transactions': results['transactions'],
        'performance': performance,
        'buyhold_returns': buyhold_returns
    }

def run_backtest(prices: pd.DataFrame, portfolio: Portfolio) -> Dict[str, Any]:
    """
    Выполнение бэктеста стратегии
    
    Args:
        prices: DataFrame с ценами
        portfolio: Экземпляр класса Portfolio для выбора активов и весов
        
    Returns:
        Словарь с результатами бэктеста
    """
    logger.info("Выполнение бэктеста умной бета-стратегии")
    
    # Результаты
    portfolio_returns = []
    portfolio_weights_history = []
    transactions_history = []
    
    # Период для расчета метрик
    lookback = LOOKBACK_PERIOD
    
    # Частота ребалансировки
    rebalance_freq = REBALANCE_FREQUENCY
    
    # Текущие веса портфеля (изначально нет позиций)
    current_weights = pd.Series(0, index=prices.columns)
    
    # Запуск бэктеста
    for i in range(lookback, len(prices), rebalance_freq):
        logger.info(f"Ребалансировка на день {prices.index[i].date()}")
        
        # Исторические данные для выбора и аллокации
        historical_prices = prices.iloc[i-lookback:i]
        
        # Выбор активов
        selected_assets = portfolio.select(historical_prices)
        
        # Аллокация весов (оптимизация по Sharpe)
        target_weights = portfolio.allocate(
            historical_prices, 
            selected_assets, 
            method="optimize",
            target_metric="sharpe"
        )
        
        # Расчет транзакций и затрат
        transactions = calculate_transactions(current_weights, target_weights)
        transaction_costs = calculate_transaction_costs(transactions, prices.iloc[i])
        
        # Сохранение весов и транзакций
        portfolio_weights_history.append({
            'date': prices.index[i],
            'weights': target_weights.to_dict()
        })
        
        transactions_history.append({
            'date': prices.index[i],
            'transactions': transactions.to_dict(),
            'costs': transaction_costs
        })
        
        # Обновление текущих весов
        current_weights = target_weights.copy()
        
        # Расчет доходности вперед
        if i + rebalance_freq <= len(prices):
            forward_prices = prices.iloc[i:i+rebalance_freq]
            forward_returns = forward_prices.pct_change().fillna(0)
            
            # Расчет дневной доходности портфеля
            daily_returns = []
            for day in range(len(forward_returns)):
                # Доходность портфеля за день
                day_return = 0
                for asset, weight in target_weights.items():
                    if asset in forward_returns.columns:
                        day_return += weight * forward_returns.iloc[day][asset]
                
                # Учет затрат на транзакции (распределенных на период ребалансировки)
                if day == 0:  # Учитываем затраты только в первый день после ребалансировки
                    day_return -= transaction_costs
                
                daily_returns.append(day_return)
            
            portfolio_returns.extend(daily_returns)
    
    # Создание временного ряда доходностей
    strategy_dates = prices.index[lookback:lookback+len(portfolio_returns)]
    strategy_returns = pd.Series(portfolio_returns, index=strategy_dates)
    
    # Расчет кумулятивной доходности
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    
    return {
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_weights': portfolio_weights_history,
        'transactions': transactions_history,
    }

def calculate_transactions(current_weights: pd.Series, target_weights: pd.Series) -> pd.Series:
    """
    Расчет необходимых транзакций для достижения целевых весов
    
    Args:
        current_weights: Текущие веса портфеля
        target_weights: Целевые веса портфеля
        
    Returns:
        Series с величинами транзакций (положительные - покупки, отрицательные - продажи)
    """
    # Объединение индексов для учета всех активов
    all_assets = list(set(current_weights.index) | set(target_weights.index))
    
    # Инициализация с нулями для отсутствующих активы
    current_full = pd.Series(0, index=all_assets)
    target_full = pd.Series(0, index=all_assets)
    
    # Заполнение известными значениями
    current_full.update(current_weights)
    target_full.update(target_weights)
    
    # Расчет разницы (положительные значения - покупки, отрицательные - продажи)
    transactions = target_full - current_full
    
    return transactions

def calculate_transaction_costs(transactions: pd.Series, prices: pd.Series) -> float:
    """
    Расчет затрат на транзакции
    
    Args:
        transactions: Series с объемами транзакций
        prices: Series с текущими ценами
        
    Returns:
        Общие затраты на транзакции
    """
    # Абсолютные значения транзакций (суммарный оборот)
    turnover = abs(transactions).sum()
    
    # Расчет затрат (комиссия в процентах)
    costs = turnover * COMMISSION_RATE
    
    return costs

def calculate_buyhold_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Расчет доходности стратегии buy-and-hold для индекса
    
    Args:
        prices: DataFrame с ценами
        
    Returns:
        Series с кумулятивной доходностью buy-and-hold
    """
    # Расчет приблизительного индекса (равновзвешенный)
    index_value = prices.mean(axis=1)
    
    # Расчет дневной доходности
    index_returns = index_value.pct_change().fillna(0)
    
    # Кумулятивная доходность
    buyhold_cumulative = (1 + index_returns).cumprod() - 1
    
    return buyhold_cumulative

def calculate_performance_metrics(results: Dict[str, Any], buyhold_returns: pd.Series) -> pd.DataFrame:
    """
    Расчет метрик эффективности стратегии
    
    Args:
        results: Словарь с результатами бэктеста
        buyhold_returns: Series с кумулятивной доходностью buy-and-hold
        
    Returns:
        DataFrame с метриками эффективности
    """
    strategy_returns = results['strategy_returns']
    cumulative_returns = results['cumulative_returns']
    
    # Создание DataFrame для метрик
    metrics = pd.DataFrame(index=['Умная Бета-Стратегия', 'Buy-and-Hold'])
    
    # Полная доходность
    metrics['Общая доходность, %'] = [
        cumulative_returns.iloc[-1] * 100,
        buyhold_returns.iloc[-1] * 100
    ]
    
    # Среднегодовая доходность
    days = len(strategy_returns)
    years = days / 252
    metrics['Среднегодовая доходность, %'] = [
        ((1 + cumulative_returns.iloc[-1]) ** (1/years) - 1) * 100,
        ((1 + buyhold_returns.iloc[-1]) ** (1/years) - 1) * 100
    ]
    
    # Волатильность
    strategy_volatility = strategy_returns.std() * np.sqrt(252) * 100
    buyhold_volatility = buyhold_returns.diff().std() * np.sqrt(252) * 100
    metrics['Годовая волатильность, %'] = [strategy_volatility, buyhold_volatility]
    
    # Коэффициент Шарпа (предполагаем безрисковую ставку 3%)
    risk_free_rate = 0.03
    metrics['Коэффициент Шарпа'] = [
        (metrics['Среднегодовая доходность, %'][0]/100 - risk_free_rate) / (strategy_volatility/100),
        (metrics['Среднегодовая доходность, %'][1]/100 - risk_free_rate) / (buyhold_volatility/100)
    ]
    
    # Максимальная просадка
    strategy_dd = calculate_max_drawdown(cumulative_returns) * 100
    buyhold_dd = calculate_max_drawdown(buyhold_returns) * 100
    metrics['Максимальная просадка, %'] = [strategy_dd, buyhold_dd]
    
    # Коэффициент Сортино (учитывает только отрицательные доходности)
    neg_strategy_returns = strategy_returns.copy()
    neg_strategy_returns[neg_strategy_returns > 0] = 0
    neg_buyhold_returns = buyhold_returns.diff().copy()
    neg_buyhold_returns[neg_buyhold_returns > 0] = 0
    
    downside_dev_strategy = neg_strategy_returns.std() * np.sqrt(252) * 100
    downside_dev_buyhold = neg_buyhold_returns.std() * np.sqrt(252) * 100
    
    metrics['Коэффициент Сортино'] = [
        (metrics['Среднегодовая доходность, %'][0]/100 - risk_free_rate) / (downside_dev_strategy/100),
        (metrics['Среднегодовая доходность, %'][1]/100 - risk_free_rate) / (downside_dev_buyhold/100)
    ]
    
    # Количество транзакций
    metrics['Количество ребалансировок'] = [
        len(results['transactions']),
        "N/A"
    ]
    
    # Средние затраты на транзакции
    total_costs = sum(t['costs'] for t in results['transactions'])
    metrics['Общие комиссии, %'] = [
        total_costs * 100,
        "N/A"
    ]
    
    return metrics

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Расчет максимальной просадки
    
    Args:
        cumulative_returns: Series с кумулятивной доходностью
        
    Returns:
        Максимальная просадка в виде десятичной дроби
    """
    # Добавление единицы для расчета на основе стоимости портфеля, а не доходности
    wealth_index = 1 + cumulative_returns
    
    # Расчет исторических максимумов
    running_max = wealth_index.cummax()
    
    # Расчет просадок
    drawdowns = wealth_index / running_max - 1
    
    # Максимальная просадка
    max_drawdown = drawdowns.min()
    
    return max_drawdown

def plot_results(results: Dict[str, Any], buyhold_returns: pd.Series) -> None:
    """
    Построение графиков результатов бэктеста
    
    Args:
        results: Словарь с результатами бэктеста
        buyhold_returns: Series с кумулятивной доходностью buy-and-hold
    """
    # Настройка стиля
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 16))
    
    # График 1: Сравнение кумулятивной доходности
    plt.subplot(3, 1, 1)
    plt.plot(results['cumulative_returns'] * 100, label='Умная Бета-Стратегия', linewidth=2)
    plt.plot(buyhold_returns * 100, label='Buy-and-Hold (равновзвешенный)', linewidth=2, linestyle='--')
    plt.title('Сравнение кумулятивной доходности', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Доходность, %')
    plt.legend()
    plt.grid(True)
    
    # График 2: Распределение доходностей
    plt.subplot(3, 1, 2)
    sns.histplot(results['strategy_returns'] * 100, kde=True, label='Умная Бета-Стратегия')
    sns.histplot(buyhold_returns.diff().dropna() * 100, kde=True, label='Buy-and-Hold')
    plt.title('Распределение дневных доходностей', fontsize=14)
    plt.xlabel('Дневная доходность, %')
    plt.ylabel('Частота')
    plt.legend()
    
    # График 3: Изменение весов во времени
    plt.subplot(3, 1, 3)
    weights_df = pd.DataFrame()
    
    # Преобразование весов в DataFrame
    for period in results['portfolio_weights']:
        date = period['date']
        weights = period['weights']
        weights_series = pd.Series(weights, name=date)
        weights_df = pd.concat([weights_df, weights_series.to_frame().T])
    
    # Отбор топ-10 акций по среднему весу для лучшей визуализации
    if not weights_df.empty:
        top_assets = weights_df.mean().nlargest(10).index
        weights_df_top = weights_df[top_assets]
        weights_df_top.plot(kind='area', stacked=True, colormap='viridis')
        plt.title('Изменение весов портфеля (топ-10 акций)', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Доля в портфеле')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # Сохранение графиков
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/smart_beta_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Графики сохранены в results/smart_beta_results.png")
    
    # Построение графика просадок
    plt.figure(figsize=(12, 6))
    
    # Расчет просадок
    wealth_index_strategy = 1 + results['cumulative_returns']
    running_max_strategy = wealth_index_strategy.cummax()
    drawdowns_strategy = (wealth_index_strategy / running_max_strategy - 1) * 100
    
    wealth_index_buyhold = 1 + buyhold_returns
    running_max_buyhold = wealth_index_buyhold.cummax()
    drawdowns_buyhold = (wealth_index_buyhold / running_max_buyhold - 1) * 100
    
    plt.plot(drawdowns_strategy, label='Умная Бета-Стратегия', linewidth=2)
    plt.plot(drawdowns_buyhold, label='Buy-and-Hold', linewidth=2, linestyle='--')
    plt.title('Сравнение просадок', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Просадка, %')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/drawdowns.png", dpi=300)
    plt.close()
    
    logger.info("График просадок сохранен в results/drawdowns.png")

def export_to_excel(results: Dict[str, Any]) -> None:
    """
    Экспорт результатов бэктеста в Excel
    
    Args:
        results: Словарь с результатами бэктеста
    """
    logger.info("Экспорт результатов в Excel")
    
    # Создание директории для результатов
    os.makedirs("results", exist_ok=True)
    
    # Создание Excel файла
    excel_path = "results/smart_beta_results.xlsx"
    
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # Экспорт цен
        results['prices'].to_excel(writer, sheet_name="Цены")
        
        # Экспорт факторов
        results['factors'].to_excel(writer, sheet_name="Факторы")
        
        # Экспорт доходностей
        pd.DataFrame({
            'Умная Бета-Стратегия': results['strategy_returns'],
            'Buy-and-Hold': results['buyhold_returns'].diff()
        }).to_excel(writer, sheet_name="Доходности")
        
        # Экспорт кумулятивных доходностей
        pd.DataFrame({
            'Умная Бета-Стратегия': results['cumulative_returns'],
            'Buy-and-Hold': results['buyhold_returns']
        }).to_excel(writer, sheet_name="Кумулятивная доходность")
        
        # Экспорт весов портфеля
        weights_df = pd.DataFrame()
        for period in results['portfolio_weights']:
            date = period['date']
            weights = period['weights']
            weights_series = pd.Series(weights, name=date)
            weights_df = pd.concat([weights_df, weights_series.to_frame().T])
        weights_df.to_excel(writer, sheet_name="Веса портфеля")
        
        # Экспорт транзакций
        transactions_df = pd.DataFrame([
            {
                'Дата': t['date'],
                'Комиссии, %': t['costs'] * 100,
                'Оборот': sum(abs(val) for val in t['transactions'].values())
            } for t in results['transactions']
        ])
        transactions_df.to_excel(writer, sheet_name="Транзакции")
        
        # Экспорт метрик эффективности
        results['performance'].to_excel(writer, sheet_name="Метрики")
        
        # Создание сводной информации
        summary = pd.DataFrame({
            'Метрика': [
                'Период бэктеста',
                'Начальная дата',
                'Конечная дата',
                'Количество акций',
                'Частота ребалансировки',
                'Комиссия за сделку, %'
            ],
            'Значение': [
                f"{len(results['strategy_returns'])} дней ({len(results['strategy_returns'])/252:.2f} лет)",
                results['strategy_returns'].index[0].strftime('%Y-%m-%d'),
                results['strategy_returns'].index[-1].strftime('%Y-%m-%d'),
                len(results['prices'].columns),
                f"{REBALANCE_FREQUENCY} дней (~1 месяц)",
                f"{COMMISSION_RATE * 100:.2f}%"
            ]
        })
        summary.to_excel(writer, sheet_name="Информация", index=False)
    
    logger.info(f"Результаты экспортированы в {excel_path}")

async def main():
    """Основная точка входа в программу"""
    logger.info("Запуск умной бета-стратегии")
    
    # Запуск стратегии на исторических данных
    # По умолчанию используются последние 2 года
    results = await run_smart_beta_strategy()
    
    # Вывод основных метрик эффективности
    print("\nРезультаты умной бета-стратегии:\n")
    print(results['performance'].to_string())
    
    # Экспорт результатов в Excel
    export_to_excel(results)
    
    logger.info("Выполнение умной бета-стратегии завершено")

if __name__ == "__main__":
    asyncio.run(main()) 