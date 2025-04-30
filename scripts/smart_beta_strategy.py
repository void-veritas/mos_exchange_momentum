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
# Создаем директорию для логов, если ее нет
os.makedirs("logs", exist_ok=True)

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Обработчик для файла
file_handler = logging.FileHandler('logs/backtest_decisions.log', mode='w')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Обработчик для детального файла решений
detailed_handler = logging.FileHandler('logs/backtest_detailed.log', mode='w')
detailed_handler.setLevel(logging.DEBUG)
detailed_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
detailed_handler.setFormatter(detailed_format)

# Добавляем обработчики в логгер
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.addHandler(detailed_handler)

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

def plot_decision_visualizations(results: Dict[str, Any], output_dir: str = "results") -> None:
    """
    Создает расширенные визуализации процесса принятия решений
    
    Args:
        results: Словарь с результатами бэктеста
        output_dir: Директория для сохранения визуализаций
    """
    logger.info("Создание расширенных визуализаций процесса принятия решений")
    
    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Визуализация изменения весов активов во времени
    plt.figure(figsize=(14, 8))
    
    # Получение данных о весах
    weights_df = pd.DataFrame()
    for period in results['portfolio_weights']:
        date = period['date']
        weights = period['weights']
        weights_series = pd.Series(weights, name=date)
        weights_df = pd.concat([weights_df, weights_series.to_frame().T])
    
    # Отбор топ-10 активов по среднему весу
    top_assets = weights_df.mean().nlargest(10).index
    weights_df_top = weights_df[top_assets]
    
    # График изменения весов
    ax = weights_df_top.plot(kind='area', stacked=True, colormap='viridis', alpha=0.7)
    plt.title('Изменение весов топ-10 активов портфеля', fontsize=16)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Вес в портфеле', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Добавление вертикальных линий для дат ребалансировки
    for period in results['transactions']:
        date = period['date']
        plt.axvline(x=date, color='r', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Визуализация оборота на каждой ребалансировке
    turnover_data = []
    for period in results['transactions']:
        date = period['date']
        transactions = period['transactions']
        turnover = sum(abs(val) for val in transactions.values())
        turnover_data.append({
            'date': date,
            'turnover': turnover,
            'costs': period['costs']
        })
    
    turnover_df = pd.DataFrame(turnover_data)
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # График оборота
    bars = ax1.bar(turnover_df['date'], turnover_df['turnover'], color='steelblue', alpha=0.7, label='Оборот')
    ax1.set_xlabel('Дата ребалансировки', fontsize=12)
    ax1.set_ylabel('Оборот (сумма весов)', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # График комиссий
    line = ax2.plot(turnover_df['date'], turnover_df['costs']*100, color='indianred', label='Комиссии (%)', marker='o')
    ax2.set_ylabel('Комиссии (%)', fontsize=12, color='indianred')
    ax2.tick_params(axis='y', labelcolor='indianred')
    
    # Легенда
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.title('Оборот и комиссии на каждой ребалансировке', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/turnover_and_costs.png", dpi=300)
    plt.close()
    
    # 3. Визуализация доходности по периодам ребалансировки
    
    # Подготовка данных о доходности по периодам
    period_returns = []
    strategy_returns = results['strategy_returns']
    
    for i in range(len(results['transactions'])):
        start_date = results['transactions'][i]['date']
        
        # Определение конечной даты (либо следующая ребалансировка, либо конец периода)
        if i < len(results['transactions']) - 1:
            end_date = results['transactions'][i+1]['date']
        else:
            end_date = strategy_returns.index[-1]
        
        # Фильтрация доходностей за период
        period_slice = strategy_returns[(strategy_returns.index >= start_date) & (strategy_returns.index < end_date)]
        
        if not period_slice.empty:
            cumulative_return = (1 + period_slice).prod() - 1
            period_returns.append({
                'start_date': start_date,
                'end_date': end_date,
                'return': cumulative_return,
                'days': len(period_slice)
            })
    
    period_returns_df = pd.DataFrame(period_returns)
    
    if not period_returns_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Создание названий для оси X (периоды ребалансировки)
        x_labels = [f"{row['start_date'].strftime('%Y-%m-%d')}" for _, row in period_returns_df.iterrows()]
        
        # График доходностей по периодам
        colors = ['green' if ret >= 0 else 'red' for ret in period_returns_df['return']]
        plt.bar(x_labels, period_returns_df['return']*100, color=colors, alpha=0.7)
        
        plt.title('Доходность портфеля по периодам между ребалансировками', fontsize=16)
        plt.xlabel('Дата начала периода', fontsize=12)
        plt.ylabel('Доходность за период (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Добавление значений над столбцами
        for i, v in enumerate(period_returns_df['return']*100):
            plt.text(i, v + np.sign(v)*0.5, f"{v:.2f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/period_returns.png", dpi=300)
        plt.close()
    
    # 4. Сравнительная визуализация бенчмарка и стратегии с указанием точек ребалансировки
    plt.figure(figsize=(14, 7))
    
    # График кумулятивной доходности
    plt.plot(results['cumulative_returns']*100, label='Умная Бета-Стратегия', linewidth=2, color='royalblue')
    plt.plot(results['buyhold_returns']*100, label='Buy-and-Hold', linewidth=2, linestyle='--', color='gray')
    
    # Добавление вертикальных линий и маркеров в точках ребалансировки
    for i, period in enumerate(results['transactions']):
        date = period['date']
        # Находим значение доходности на эту дату
        if date in results['cumulative_returns'].index:
            ret_value = results['cumulative_returns'].loc[date] * 100
            plt.scatter([date], [ret_value], color='red', s=50, zorder=5)
            
            # Добавляем номер ребалансировки для первых нескольких точек
            if i < 5:  # Ограничиваем количество аннотаций для читаемости
                plt.annotate(f"#{i+1}", 
                             (date, ret_value),
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.axvline(x=date, color='lightcoral', linestyle=':', alpha=0.4)
    
    plt.title('Сравнение кумулятивной доходности стратегии и бенчмарка', fontsize=16)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Кумулятивная доходность (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_vs_benchmark_detailed.png", dpi=300)
    plt.close()
    
    # 5. Визуализация изменения количества активов в портфеле
    asset_counts = []
    
    for period in results['portfolio_weights']:
        date = period['date']
        weights = period['weights']
        # Подсчет активов с весом > 1%
        significant_assets = sum(1 for w in weights.values() if w > 0.01)
        # Подсчет активов с любым положительным весом
        all_assets = sum(1 for w in weights.values() if w > 0)
        
        asset_counts.append({
            'date': date,
            'significant_assets': significant_assets,
            'all_assets': all_assets
        })
    
    asset_counts_df = pd.DataFrame(asset_counts)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(asset_counts_df['date'], asset_counts_df['all_assets'], 
             label='Все активы', marker='o', linestyle='-', color='royalblue')
    plt.plot(asset_counts_df['date'], asset_counts_df['significant_assets'], 
             label='Значимые активы (>1%)', marker='s', linestyle='-', color='darkorange')
    
    plt.title('Изменение количества активов в портфеле', fontsize=16)
    plt.xlabel('Дата ребалансировки', fontsize=12)
    plt.ylabel('Количество активов', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/asset_count_changes.png", dpi=300)
    plt.close()
    
    logger.info(f"Расширенные визуализации сохранены в директории: {output_dir}")

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
    
    # Построение расширенных визуализаций
    plot_decision_visualizations(
        {**results, 'buyhold_returns': buyhold_returns, 'prices': prices, 'factors': factors}
    )
    
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
    logger.info(f"Параметры бэктеста: комиссия={COMMISSION_RATE*100}%, "
               f"ребалансировка каждые {REBALANCE_FREQUENCY} дней, "
               f"период данных для анализа {LOOKBACK_PERIOD} дней")
    
    # Результаты
    portfolio_returns = []
    portfolio_weights_history = []
    transactions_history = []
    
    # Правильное разделение данных на выборки
    # Разделение на тренировочную и тестовую выборки с учетом cooling периода
    train_end_idx = int(len(prices) * 0.6)  # Используем 60% для тренировки
    cooling_period = REBALANCE_FREQUENCY  # Один период ребалансировки как cooling период
    test_start_idx = train_end_idx + cooling_period

    # Отображение информации о разделении данных
    logger.info(f"Разделение данных: Обучение ({prices.index[0].date()} до {prices.index[train_end_idx-1].date()}), "
               f"Период охлаждения ({prices.index[train_end_idx].date()} до {prices.index[test_start_idx-1].date()}), "
               f"Тестирование ({prices.index[test_start_idx].date()} до {prices.index[-1].date()})")
    
    # Тренировочные данные для настройки стратегии
    train_prices = prices.iloc[:train_end_idx]
    logger.info(f"Настройка стратегии на тренировочных данных ({len(train_prices)} точек)")
    
    # Период для расчета метрик
    lookback = LOOKBACK_PERIOD
    
    # Частота ребалансировки
    rebalance_freq = REBALANCE_FREQUENCY
    
    # ========= ЭТАП 1: Настройка стратегии на тренировочных данных =========
    # Этот этап только для настройки параметров, не для оценки результатов
    for i in range(lookback, len(train_prices), rebalance_freq):
        rebalance_date = train_prices.index[i].date()
        logger.info(f"===== [ТРЕНИРОВКА] Ребалансировка {rebalance_date} =====")
        
        # Используем исторические данные для выбора активов
        historical_prices = train_prices.iloc[i-lookback:i]
        
        # Выбор активов
        selected_assets = portfolio.select(historical_prices)
        logger.debug(f"[ТРЕНИРОВКА] Выбрано {len(selected_assets)} активов: {', '.join(selected_assets)}")
        
        # Аллокация весов
        _ = portfolio.allocate(
            historical_prices, 
            selected_assets, 
            method="optimize",
            target_metric="sharpe"
        )
    
    logger.info("Настройка стратегии завершена")
    
    # ========= ЭТАП 2: Тестирование на тестовой выборке =========
    # Тестовые данные для оценки стратегии (после cooling периода)
    test_prices = prices.iloc[test_start_idx:]
    logger.info(f"Тестирование стратегии на тестовых данных ({len(test_prices)} точек)")
    
    # Текущие веса портфеля (изначально нет позиций)
    current_weights = pd.Series(0, index=prices.columns)
    
    # Запуск бэктеста на тестовой выборке
    for i in range(lookback, len(test_prices), rebalance_freq):
        rebalance_date = test_prices.index[i].date()
        logger.info(f"====== [ТЕСТ] Ребалансировка {rebalance_date} ======")
        logger.debug(f"Детальный лог принятия решений для ребалансировки {rebalance_date}")
        
        # Важно: используем полную историю до текущего момента (включая тренировочные данные)
        # Это имитирует реальную ситуацию, когда у нас есть вся предыдущая история
        full_idx = test_start_idx + i
        current_date = test_prices.index[i].date()
        
        # Получаем исторические данные с учетом lookback периода
        if full_idx < lookback:
            # Если недостаточно данных, используем все доступные
            historical_prices = prices.iloc[:full_idx]
        else:
            # Используем последние lookback точек
            historical_prices = prices.iloc[full_idx-lookback:full_idx]
        
        # Логирование статистики исторических данных
        returns = historical_prices.pct_change().dropna()
        logger.debug(f"Статистика доходности за период анализа ({historical_prices.index[0].date()} - {historical_prices.index[-1].date()}):")
        logger.debug(f"Средняя дневная доходность: \n{returns.mean()}")
        logger.debug(f"Волатильность (СКО): \n{returns.std()}")
        logger.debug(f"Общая доходность за период: \n{(historical_prices.iloc[-1] / historical_prices.iloc[0] - 1) * 100}%")
        
        # Выбор активов
        logger.info(f"Выбор активов методом {portfolio.select_by}, оптимизация по {portfolio.select_metric}")
        selected_assets = portfolio.select(historical_prices)
        logger.info(f"Выбрано {len(selected_assets)} активов: {', '.join(selected_assets)}")
        
        # Логирование метрик выбранных активов
        metrics = calculate_asset_metrics(historical_prices, selected_assets)
        logger.debug("Метрики выбранных активов:")
        for asset, asset_metrics in metrics.items():
            logger.debug(f"{asset}: Доходность={asset_metrics['return']:.2%}, "
                        f"Волатильность={asset_metrics['volatility']:.2%}, "
                        f"Шарп={asset_metrics['sharpe']:.2f}, "
                        f"Макс. просадка={asset_metrics['max_drawdown']:.2%}")
        
        # Аллокация весов (оптимизация по Sharpe)
        logger.info(f"Аллокация весов методом 'optimize', целевая метрика 'sharpe'")
        target_weights = portfolio.allocate(
            historical_prices, 
            selected_assets, 
            method="optimize",
            target_metric="sharpe"
        )
        
        # Логирование целевых весов
        logger.info("Рассчитанные целевые веса портфеля:")
        for asset, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
            logger.info(f"  {asset}: {weight:.2%}")
        
        # Расчет транзакций и затрат
        transactions = calculate_transactions(current_weights, target_weights)
        transaction_costs = calculate_transaction_costs(transactions, test_prices.iloc[i])
        
        # Логирование транзакций
        logger.info("Транзакции для достижения целевых весов:")
        buys = [(ticker, value) for ticker, value in transactions.items() if value > 0]
        sells = [(ticker, value) for ticker, value in transactions.items() if value < 0]
        
        logger.info("Покупки:")
        for ticker, value in sorted(buys, key=lambda x: -x[1]):
            logger.info(f"  {ticker}: +{value:.4f}")
        
        logger.info("Продажи:")
        for ticker, value in sorted(sells, key=lambda x: x[1]):
            logger.info(f"  {ticker}: {value:.4f}")
        
        logger.info(f"Общий оборот: {abs(transactions).sum():.4f}")
        logger.info(f"Комиссионные затраты: {transaction_costs:.4f} ({transaction_costs*100:.2f}%)")
        
        # Сохранение весов и транзакций
        portfolio_weights_history.append({
            'date': test_prices.index[i],
            'weights': target_weights.to_dict()
        })
        
        transactions_history.append({
            'date': test_prices.index[i],
            'transactions': transactions.to_dict(),
            'costs': transaction_costs
        })
        
        # Обновление текущих весов
        current_weights = target_weights.copy()
        
        # Расчет доходности вперед
        if i + rebalance_freq <= len(test_prices):
            forward_prices = test_prices.iloc[i:i+rebalance_freq]
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
            
            # Логирование ожидаемой будущей доходности
            expected_return = 0
            for asset, weight in target_weights.items():
                expected_return += weight * returns[asset].mean() * rebalance_freq
            
            logger.info(f"Ожидаемая доходность портфеля на следующий период: {expected_return:.2%}")
            
            portfolio_returns.extend(daily_returns)
            
            # Логирование фактических результатов
            actual_return = sum(daily_returns)
            logger.info(f"Фактическая доходность за период после ребалансировки: {actual_return:.2%}")
            logger.info(f"Разница ожидаемой и фактической доходности: {actual_return - expected_return:.2%}")
    
    # Преобразование в Series
    returns_series = pd.Series(portfolio_returns, index=test_prices.index[lookback:lookback+len(portfolio_returns)])
    
    # Расчет кумулятивной доходности
    cumulative_returns = (1 + returns_series).cumprod() - 1
    
    return {
        'strategy_returns': returns_series,
        'cumulative_returns': cumulative_returns,
        'portfolio_weights': portfolio_weights_history,
        'transactions': transactions_history
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
        
        # Подробный лист транзакций с разбивкой по активам
        detailed_transactions = []
        for t in results['transactions']:
            for asset, value in t['transactions'].items():
                if abs(value) > 0.0001:  # Игнорируем очень маленькие транзакции
                    detailed_transactions.append({
                        'Дата': t['date'],
                        'Актив': asset,
                        'Изменение веса': value,
                        'Тип': 'Покупка' if value > 0 else 'Продажа'
                    })
        pd.DataFrame(detailed_transactions).to_excel(writer, sheet_name="Детальные транзакции")
        
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
                'Комиссия за сделку, %',
                'Общая доходность, %',
                'Годовая доходность, %',
                'Годовая волатильность, %',
                'Коэффициент Шарпа',
                'Максимальная просадка, %',
                'Общий оборот, %',
                'Количество сделок'
            ],
            'Значение': [
                f"{len(results['strategy_returns'])} дней ({len(results['strategy_returns'])/252:.2f} лет)",
                results['strategy_returns'].index[0].strftime('%Y-%m-%d'),
                results['strategy_returns'].index[-1].strftime('%Y-%m-%d'),
                len(results['prices'].columns),
                f"{REBALANCE_FREQUENCY} дней (~1 месяц)",
                f"{COMMISSION_RATE * 100:.2f}%",
                f"{results['cumulative_returns'].iloc[-1] * 100:.2f}%",
                f"{((1 + results['cumulative_returns'].iloc[-1]) ** (252/len(results['strategy_returns'])) - 1) * 100:.2f}%",
                f"{results['strategy_returns'].std() * np.sqrt(252) * 100:.2f}%",
                f"{(results['strategy_returns'].mean() / results['strategy_returns'].std()) * np.sqrt(252):.2f}",
                f"{calculate_max_drawdown(results['cumulative_returns']) * 100:.2f}%",
                f"{sum(sum(abs(val) for val in t['transactions'].values()) for t in results['transactions']) * 100:.2f}%",
                len(detailed_transactions)
            ]
        })
        summary.to_excel(writer, sheet_name="Информация", index=False)
    
    logger.info(f"Результаты экспортированы в {excel_path}")
    
    # Также сохраним лог-файл как отдельный лист в следующем запуске
    log_path = "logs/backtest_decisions.log"
    if os.path.exists(log_path):
        logger.info(f"Лог-файл с решениями доступен по пути: {log_path}")

def calculate_asset_metrics(prices: pd.DataFrame, assets: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Расчет метрик для выбранных активов
    
    Args:
        prices: DataFrame с ценами
        assets: Список выбранных активов
        
    Returns:
        Словарь с метриками для каждого актива
    """
    # Фильтрация только выбранных активов
    asset_prices = prices[assets]
    
    # Расчет доходностей
    returns = asset_prices.pct_change().dropna()
    
    # Создание словаря для хранения метрик
    metrics = {}
    
    # Расчет метрик для каждого актива
    for asset in assets:
        asset_return = returns[asset]
        
        # Общая доходность
        total_return = (asset_prices[asset].iloc[-1] / asset_prices[asset].iloc[0]) - 1
        
        # Волатильность
        volatility = asset_return.std() * np.sqrt(252)
        
        # Коэффициент Шарпа (предполагаем безрисковую ставку 0 для простоты)
        sharpe = (asset_return.mean() / asset_return.std()) * np.sqrt(252) if asset_return.std() > 0 else 0
        
        # Максимальная просадка
        cum_returns = (1 + asset_return).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        metrics[asset] = {
            'return': total_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    return metrics

def export_logs_to_excel(log_file_path: str, output_path: str) -> None:
    """
    Экспортирует лог-файл в Excel с более удобным форматированием
    
    Args:
        log_file_path: Путь к лог-файлу
        output_path: Путь для сохранения Excel файла
    """
    logger.info(f"Экспорт логов в Excel: {output_path}")
    
    # Чтение лога
    if not os.path.exists(log_file_path):
        logger.warning(f"Лог-файл не найден: {log_file_path}")
        return
    
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()
    
    # Парсинг логов и формирование структурированных данных
    rebalance_dates = []
    current_date = None
    asset_selections = []
    weight_allocations = []
    transaction_details = []
    performance_metrics = []
    
    for line in log_lines:
        if "====== Ребалансировка" in line:
            # Новая дата ребалансировки
            date_str = line.split("Ребалансировка")[1].strip().split()[0]
            current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            rebalance_dates.append(current_date)
        
        if current_date and "Выбрано" in line and "активов" in line:
            # Информация о выбранных активах
            selected = line.split("Выбрано")[1].split("активов:")[1].strip()
            asset_selections.append({
                'date': current_date,
                'selected_assets': selected
            })
        
        if current_date and "Рассчитанные целевые веса портфеля:" in line:
            # Начало блока с весами
            weight_data = {
                'date': current_date,
                'weights': {}
            }
            weight_allocations.append(weight_data)
        
        # Строки с весами активов
        if current_date and weight_allocations and line.strip().startswith("  ") and ":" in line:
            parts = line.strip().split(":")
            if len(parts) == 2 and weight_allocations[-1]['date'] == current_date:
                asset = parts[0].strip()
                weight = parts[1].strip()
                if "%" in weight:  # Проверяем формат (с % или без)
                    weight = float(weight.replace("%", "")) / 100
                weight_allocations[-1]['weights'][asset] = weight
        
        if current_date and "Транзакции для достижения целевых весов:" in line:
            # Начало блока с транзакциями
            transaction_data = {
                'date': current_date,
                'buys': {},
                'sells': {}
            }
            transaction_details.append(transaction_data)
        
        # Строки с покупками
        if current_date and transaction_details and "Покупки:" in line:
            mode = "buys"
        
        # Строки с продажами
        if current_date and transaction_details and "Продажи:" in line:
            mode = "sells"
        
        # Парсинг транзакций
        if current_date and transaction_details and line.strip().startswith("  ") and ":" in line and ("+") in line or ("-") in line:
            parts = line.strip().split(":")
            if len(parts) == 2 and transaction_details[-1]['date'] == current_date:
                asset = parts[0].strip()
                value = float(parts[1].strip())
                if "buys" in locals() and mode == "buys":
                    transaction_details[-1]['buys'][asset] = value
                elif "sells" in locals() and mode == "sells":
                    transaction_details[-1]['sells'][asset] = value
        
        if current_date and "Фактическая доходность за период" in line:
            # Доходность за период
            return_str = line.split("Фактическая доходность за период")[1].split(":")[1].strip()
            if "%" in return_str:
                return_value = float(return_str.replace("%", "")) / 100
            else:
                return_value = float(return_str)
            
            performance_metrics.append({
                'date': current_date,
                'actual_return': return_value
            })
    
    # Создание DataFrame для каждого типа данных
    rebalance_df = pd.DataFrame(rebalance_dates, columns=['Дата ребалансировки'])
    
    # Данные о выбранных активах
    asset_selection_df = pd.DataFrame(asset_selections)
    
    # Данные о весах портфеля
    weights_data = []
    for allocation in weight_allocations:
        for asset, weight in allocation['weights'].items():
            weights_data.append({
                'Дата': allocation['date'],
                'Актив': asset,
                'Вес': weight
            })
    weights_df = pd.DataFrame(weights_data)
    
    # Данные о транзакциях
    transaction_data = []
    for trans in transaction_details:
        # Покупки
        for asset, value in trans.get('buys', {}).items():
            transaction_data.append({
                'Дата': trans['date'],
                'Актив': asset,
                'Тип': 'Покупка',
                'Значение': value
            })
        # Продажи
        for asset, value in trans.get('sells', {}).items():
            transaction_data.append({
                'Дата': trans['date'],
                'Актив': asset,
                'Тип': 'Продажа',
                'Значение': value
            })
    
    transactions_df = pd.DataFrame(transaction_data)
    
    # Данные о доходности
    performance_df = pd.DataFrame(performance_metrics)
    
    # Сохранение в Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        rebalance_df.to_excel(writer, sheet_name='Даты ребалансировки', index=False)
        asset_selection_df.to_excel(writer, sheet_name='Выбранные активы', index=False)
        weights_df.to_excel(writer, sheet_name='Веса портфеля', index=False)
        transactions_df.to_excel(writer, sheet_name='Транзакции', index=False)
        if not performance_df.empty:
            performance_df.to_excel(writer, sheet_name='Доходность', index=False)
        
        # Сырые логи
        log_df = pd.DataFrame(log_lines, columns=['Лог'])
        log_df.to_excel(writer, sheet_name='Сырые логи', index=False)
    
    logger.info(f"Логи успешно экспортированы в {output_path}")

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
    
    # Экспорт логов в Excel
    log_path = "logs/backtest_decisions.log"
    excel_log_path = "results/backtest_decisions.xlsx"
    export_logs_to_excel(log_path, excel_log_path)
    
    # Сообщение о логах
    log_path = os.path.abspath("logs/backtest_decisions.log")
    detailed_log_path = os.path.abspath("logs/backtest_detailed.log")
    excel_log_path = os.path.abspath("results/backtest_decisions.xlsx")
    print(f"\nПодробный лог принятия решений сохранен в:")
    print(f" - Текстовый лог: {log_path}")
    print(f" - Детальный лог: {detailed_log_path}")
    print(f" - Excel-формат: {excel_log_path}")
    
    logger.info("Выполнение умной бета-стратегии завершено")

if __name__ == "__main__":
    asyncio.run(main()) 