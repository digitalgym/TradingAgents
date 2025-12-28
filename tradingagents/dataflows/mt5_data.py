"""
MetaTrader 5 data provider for TradingAgents.
Supports commodities (gold, silver, platinum, copper), forex, and other MT5 instruments.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Annotated, Optional
from .config import get_config

# MT5 Timeframe mapping
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Commodity symbol mappings for user convenience
COMMODITY_ALIASES = {
    "gold": "XAUUSD",
    "silver": "XAGUSD",
    "platinum": "XPTUSD",
    "copper": "COPPER-C",
    "xau": "XAUUSD",
    "xag": "XAGUSD",
    "xpt": "XPTUSD",
}


def _ensure_mt5_initialized() -> bool:
    """Ensure MT5 is initialized, initialize if not."""
    if not mt5.terminal_info():
        if not mt5.initialize():
            error = mt5.last_error()
            raise ConnectionError(f"MT5 initialization failed: {error}")
    return True


def _resolve_symbol(symbol: str) -> str:
    """Resolve symbol aliases to actual MT5 symbols."""
    return COMMODITY_ALIASES.get(symbol.lower(), symbol.upper())


def get_mt5_data(
    symbol: Annotated[str, "MT5 symbol or alias (e.g., XAUUSD, gold, EURUSD)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    timeframe: Annotated[str, "Timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1"] = "D1",
) -> str:
    """
    Retrieve OHLCV data from MetaTrader 5.
    
    Args:
        symbol: MT5 symbol or alias (gold, silver, XAUUSD, EURUSD, etc.)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        timeframe: Timeframe (default D1 for daily)
    
    Returns:
        CSV string containing OHLCV data
    """
    _ensure_mt5_initialized()
    
    # Resolve symbol alias
    mt5_symbol = _resolve_symbol(symbol)
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Get timeframe constant
    tf = TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_D1)
    
    # Ensure symbol is available in Market Watch
    if not mt5.symbol_select(mt5_symbol, True):
        available = [s.name for s in mt5.symbols_get() if mt5_symbol.upper() in s.name.upper()]
        raise ValueError(f"Symbol '{mt5_symbol}' not found. Similar: {available[:5]}")
    
    # Fetch data
    rates = mt5.copy_rates_range(mt5_symbol, tf, start_dt, end_dt)
    
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        return f"No data found for symbol '{mt5_symbol}' between {start_date} and {end_date}. Error: {error}"
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'spread': 'Spread',
        'real_volume': 'RealVolume'
    })
    
    # Select relevant columns
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if 'Spread' in df.columns:
        columns.append('Spread')
    df = df[columns]
    
    # Round prices for cleaner display
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].round(5)
    
    # Convert to CSV string
    csv_string = df.to_csv(index=False)
    
    # Add header
    header = f"# MT5 data for {mt5_symbol} from {start_date} to {end_date}\n"
    header += f"# Timeframe: {timeframe}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return header + csv_string


def get_mt5_symbol_info(
    symbol: Annotated[str, "MT5 symbol or alias"],
) -> dict:
    """Get detailed information about an MT5 symbol."""
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    if not mt5.symbol_select(mt5_symbol, True):
        raise ValueError(f"Symbol '{mt5_symbol}' not found")
    
    info = mt5.symbol_info(mt5_symbol)
    if info is None:
        raise ValueError(f"Could not get info for symbol '{mt5_symbol}'")
    
    return {
        "symbol": info.name,
        "description": info.description,
        "currency_base": info.currency_base,
        "currency_profit": info.currency_profit,
        "digits": info.digits,
        "point": info.point,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "spread": info.spread,
        "bid": info.bid,
        "ask": info.ask,
    }


def get_mt5_current_price(
    symbol: Annotated[str, "MT5 symbol or alias"],
) -> dict:
    """Get current bid/ask price for a symbol."""
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    if not mt5.symbol_select(mt5_symbol, True):
        raise ValueError(f"Symbol '{mt5_symbol}' not found")
    
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        raise ValueError(f"Could not get tick for symbol '{mt5_symbol}'")
    
    return {
        "symbol": mt5_symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "volume": tick.volume,
        "time": datetime.fromtimestamp(tick.time).strftime('%Y-%m-%d %H:%M:%S'),
    }


def list_mt5_symbols(
    filter_pattern: Annotated[str, "Filter pattern (e.g., 'XAU', 'USD')"] = None,
) -> list:
    """List available MT5 symbols, optionally filtered."""
    _ensure_mt5_initialized()
    
    symbols = mt5.symbols_get()
    if symbols is None:
        return []
    
    symbol_names = [s.name for s in symbols]
    
    if filter_pattern:
        pattern = filter_pattern.upper()
        symbol_names = [s for s in symbol_names if pattern in s.upper()]
    
    return symbol_names


def get_asset_type(symbol: str) -> str:
    """
    Determine the asset type from a symbol.
    
    Returns: 'commodity', 'forex', 'crypto', 'stock', or 'unknown'
    """
    symbol_upper = symbol.upper()
    
    # Commodities
    commodity_patterns = ['XAU', 'XAG', 'XPT', 'COPPER', 'XCU', 'OIL', 'BRENT', 'WTI', 'NATGAS']
    if any(p in symbol_upper for p in commodity_patterns):
        return 'commodity'
    
    # Crypto
    crypto_patterns = ['BTC', 'ETH', 'LTC', 'XRP', 'CRYPTO', '.CRP']
    if any(p in symbol_upper for p in crypto_patterns):
        return 'crypto'
    
    # Forex (currency pairs are typically 6 chars like EURUSD)
    forex_currencies = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'SEK', 'NOK', 'SGD', 'HKD', 'CNH']
    if len(symbol_upper) == 6:
        base = symbol_upper[:3]
        quote = symbol_upper[3:]
        if base in forex_currencies and quote in forex_currencies:
            return 'forex'
    
    # Stock indices
    index_patterns = ['US500', 'US30', 'US100', 'GER40', 'UK100', 'JPN225', 'AUS200']
    if any(p in symbol_upper for p in index_patterns):
        return 'index'
    
    return 'unknown'


def shutdown_mt5():
    """Shutdown MT5 connection."""
    mt5.shutdown()
