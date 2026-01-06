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


def check_mt5_autotrading() -> dict:
    """
    Check if MT5 AutoTrading is enabled.
    
    Returns:
        dict with keys:
        - connected: bool - whether MT5 is connected
        - autotrading_enabled: bool - whether AutoTrading is enabled
        - trade_allowed: bool - whether trading is allowed for the account
        - message: str - human-readable status message
    """
    result = {
        "connected": False,
        "autotrading_enabled": False,
        "trade_allowed": False,
        "message": "",
    }
    
    try:
        if not mt5.initialize():
            result["message"] = "MT5 not connected - please open MetaTrader 5"
            return result
        
        result["connected"] = True
        
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            result["message"] = "Could not get MT5 terminal info"
            return result
        
        # Check if AutoTrading is enabled in the terminal
        result["autotrading_enabled"] = terminal_info.trade_allowed
        
        # Check account trading permissions
        account_info = mt5.account_info()
        if account_info:
            result["trade_allowed"] = account_info.trade_allowed
        
        if not result["autotrading_enabled"]:
            result["message"] = "⚠️ AutoTrading DISABLED - Click the 'AutoTrading' button in MT5 toolbar to enable"
        elif not result["trade_allowed"]:
            result["message"] = "⚠️ Trading not allowed for this account"
        else:
            result["message"] = "✓ AutoTrading enabled"
        
        return result
        
    except Exception as e:
        result["message"] = f"Error checking MT5 status: {e}"
        return result


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


def get_mt5_indicator(
    symbol: Annotated[str, "MT5 symbol or alias"],
    indicator: Annotated[str, "Indicator name: rsi, macd, bbands, sma, ema, atr, adx, stoch"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: int = 30,
    period: int = 14,
    timeframe: str = "D1",
) -> str:
    """
    Calculate technical indicators from MT5 OHLCV data.
    
    Supported indicators:
    - rsi: Relative Strength Index
    - macd: MACD (12, 26, 9)
    - bbands: Bollinger Bands (20, 2)
    - sma: Simple Moving Average
    - ema: Exponential Moving Average
    - atr: Average True Range
    - adx: Average Directional Index
    - stoch: Stochastic Oscillator
    - close_200_sma: 200-period SMA of close
    
    Returns:
        String with indicator values
    """
    import numpy as np
    from datetime import timedelta
    
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    # Handle curr_date - convert to string if needed
    if isinstance(curr_date, int):
        curr_date = str(curr_date)
    
    # Parse end date (curr_date) and calculate start date from look_back_days
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)
    
    # Get extra bars for indicator calculation warmup
    # For 200 SMA on daily, need 200+ trading days (~300 calendar days)
    warmup_days = max(400, period * 3)  # Enough for 200 SMA with buffer
    warmup_start = start_dt - timedelta(days=warmup_days)
    
    tf = TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_D1)
    
    if not mt5.symbol_select(mt5_symbol, True):
        return f"Symbol '{mt5_symbol}' not found"
    
    rates = mt5.copy_rates_range(mt5_symbol, tf, warmup_start, end_dt)
    
    if rates is None or len(rates) == 0:
        return f"No data for {mt5_symbol}"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    
    indicator = indicator.lower().replace("-", "_")
    result_lines = []
    
    # Calculate requested indicator
    if indicator == "rsi":
        rsi = _calculate_rsi(close, period)
        result_lines.append(f"RSI({period}): {rsi[-1]:.2f}")
        result_lines.append(f"RSI Previous: {rsi[-2]:.2f}" if len(rsi) > 1 else "")
        
    elif indicator == "macd":
        macd_line, signal_line, histogram = _calculate_macd(close)
        result_lines.append(f"MACD Line: {macd_line[-1]:.4f}")
        result_lines.append(f"Signal Line: {signal_line[-1]:.4f}")
        result_lines.append(f"Histogram: {histogram[-1]:.4f}")
        
    elif indicator == "bbands":
        upper, middle, lower = _calculate_bbands(close, 20, 2)
        result_lines.append(f"Upper Band: {upper[-1]:.5f}")
        result_lines.append(f"Middle Band: {middle[-1]:.5f}")
        result_lines.append(f"Lower Band: {lower[-1]:.5f}")
        result_lines.append(f"Current Price: {close[-1]:.5f}")
        
    elif indicator == "sma" or indicator == "close_sma":
        sma = _calculate_sma(close, period)
        result_lines.append(f"SMA({period}): {sma[-1]:.5f}")
        result_lines.append(f"Price: {close[-1]:.5f}")
        result_lines.append(f"Price vs SMA: {'Above' if close[-1] > sma[-1] else 'Below'}")
        
    elif indicator == "close_200_sma":
        sma = _calculate_sma(close, 200)
        result_lines.append(f"SMA(200): {sma[-1]:.5f}")
        result_lines.append(f"Price: {close[-1]:.5f}")
        result_lines.append(f"Price vs SMA200: {'Above' if close[-1] > sma[-1] else 'Below'}")
        
    elif indicator == "ema":
        ema = _calculate_ema(close, period)
        result_lines.append(f"EMA({period}): {ema[-1]:.5f}")
        
    elif indicator == "atr":
        atr = _calculate_atr(high, low, close, period)
        result_lines.append(f"ATR({period}): {atr[-1]:.5f}")
        result_lines.append(f"ATR as % of price: {(atr[-1]/close[-1])*100:.2f}%")
        
    elif indicator == "adx":
        adx, plus_di, minus_di = _calculate_adx(high, low, close, period)
        result_lines.append(f"ADX({period}): {adx[-1]:.2f}")
        result_lines.append(f"+DI: {plus_di[-1]:.2f}")
        result_lines.append(f"-DI: {minus_di[-1]:.2f}")
        trend = "Strong" if adx[-1] > 25 else "Weak"
        result_lines.append(f"Trend Strength: {trend}")
        
    elif indicator == "stoch":
        k, d = _calculate_stochastic(high, low, close, period)
        result_lines.append(f"Stoch %K: {k[-1]:.2f}")
        result_lines.append(f"Stoch %D: {d[-1]:.2f}")
        
    else:
        return f"Unknown indicator: {indicator}. Supported: rsi, macd, bbands, sma, ema, atr, adx, stoch, close_200_sma"
    
    # Add metadata
    header = f"# {indicator.upper()} for {mt5_symbol}\n"
    header += f"# Period: {period}, Timeframe: {timeframe}\n"
    header += f"# Calculated from MT5 data on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    return header + "\n".join(result_lines)


def _calculate_rsi(close: 'np.ndarray', period: int = 14) -> 'np.ndarray':
    """Calculate RSI."""
    import numpy as np
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    # Initial SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_macd(close: 'np.ndarray', fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD."""
    ema_fast = _calculate_ema(close, fast)
    ema_slow = _calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_bbands(close: 'np.ndarray', period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    import numpy as np
    middle = _calculate_sma(close, period)
    std = np.zeros(len(close))
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i-period+1:i+1])
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def _calculate_sma(data: 'np.ndarray', period: int) -> 'np.ndarray':
    """Calculate Simple Moving Average."""
    import numpy as np
    sma = np.zeros(len(data))
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i-period+1:i+1])
    return sma


def _calculate_ema(data: 'np.ndarray', period: int) -> 'np.ndarray':
    """Calculate Exponential Moving Average."""
    import numpy as np
    ema = np.zeros(len(data))
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def _calculate_atr(high: 'np.ndarray', low: 'np.ndarray', close: 'np.ndarray', period: int = 14) -> 'np.ndarray':
    """Calculate Average True Range."""
    import numpy as np
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.zeros(len(close))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def _calculate_adx(high: 'np.ndarray', low: 'np.ndarray', close: 'np.ndarray', period: int = 14) -> tuple:
    """Calculate ADX, +DI, -DI."""
    import numpy as np
    
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = _calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(len(close))
    minus_di = np.zeros(len(close))
    
    # Smooth DM
    smoothed_plus_dm = _calculate_ema(plus_dm, period)
    smoothed_minus_dm = _calculate_ema(minus_dm, period)
    
    # Calculate DI
    for i in range(period, len(close)):
        if atr[i] != 0:
            plus_di[i] = (smoothed_plus_dm[i] / atr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / atr[i]) * 100
    
    # Calculate DX and ADX
    dx = np.zeros(len(close))
    for i in range(period, len(close)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
    
    adx = _calculate_ema(dx, period)
    
    return adx, plus_di, minus_di


def _calculate_stochastic(high: 'np.ndarray', low: 'np.ndarray', close: 'np.ndarray', period: int = 14) -> tuple:
    """Calculate Stochastic Oscillator."""
    import numpy as np
    
    k = np.zeros(len(close))
    
    for i in range(period - 1, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high != lowest_low:
            k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
    
    d = _calculate_sma(k, 3)
    
    return k, d


# =============================================================================
# TRADE EXECUTION FUNCTIONS
# =============================================================================

def place_limit_order(
    symbol: str,
    order_type: str,  # "BUY_LIMIT" or "SELL_LIMIT"
    volume: float,
    price: float,
    sl: float = None,
    tp: float = None,
    comment: str = "TradingAgents",
    magic: int = 123456,
    expiration: datetime = None,
) -> dict:
    """
    Place a limit order in MT5.
    
    Args:
        symbol: MT5 symbol (e.g., XAUUSD, XAGUSD)
        order_type: "BUY_LIMIT" or "SELL_LIMIT"
        volume: Lot size (e.g., 0.01, 0.1, 1.0)
        price: Limit price for entry
        sl: Stop loss price (optional)
        tp: Take profit price (optional)
        comment: Order comment
        magic: Magic number for order identification
        expiration: Order expiration datetime (optional)
    
    Returns:
        dict with order result or error
    """
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    # Ensure symbol is available
    if not mt5.symbol_select(mt5_symbol, True):
        return {"success": False, "error": f"Symbol '{mt5_symbol}' not found"}
    
    # Get symbol info for validation
    symbol_info = mt5.symbol_info(mt5_symbol)
    if symbol_info is None:
        return {"success": False, "error": f"Could not get info for '{mt5_symbol}'"}
    
    # Validate volume
    if volume < symbol_info.volume_min:
        return {"success": False, "error": f"Volume {volume} below minimum {symbol_info.volume_min}"}
    if volume > symbol_info.volume_max:
        return {"success": False, "error": f"Volume {volume} above maximum {symbol_info.volume_max}"}
    
    # Round volume to step
    volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
    
    # Determine order type
    if order_type.upper() == "BUY_LIMIT":
        mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
    elif order_type.upper() == "SELL_LIMIT":
        mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
    elif order_type.upper() == "BUY":
        mt5_order_type = mt5.ORDER_TYPE_BUY
    elif order_type.upper() == "SELL":
        mt5_order_type = mt5.ORDER_TYPE_SELL
    else:
        return {"success": False, "error": f"Invalid order type: {order_type}"}
    
    # Build request
    request = {
        "action": mt5.TRADE_ACTION_PENDING if "LIMIT" in order_type.upper() else mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": volume,
        "type": mt5_order_type,
        "price": price,
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp
    if expiration is not None:
        request["expiration"] = int(expiration.timestamp())
        request["type_time"] = mt5.ORDER_TIME_SPECIFIED
    
    # Send order
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        return {"success": False, "error": f"Order send failed: {error}"}
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "success": False,
            "error": f"Order failed: {result.comment}",
            "retcode": result.retcode,
        }
    
    return {
        "success": True,
        "order_id": result.order,
        "deal_id": result.deal,
        "volume": result.volume,
        "price": result.price,
        "comment": result.comment,
    }


def place_market_order(
    symbol: str,
    order_type: str,  # "BUY" or "SELL"
    volume: float,
    sl: float = None,
    tp: float = None,
    comment: str = "TradingAgents",
    magic: int = 123456,
) -> dict:
    """
    Place a market order in MT5.
    
    Args:
        symbol: MT5 symbol (e.g., XAUUSD, XAGUSD)
        order_type: "BUY" or "SELL"
        volume: Lot size
        sl: Stop loss price (optional)
        tp: Take profit price (optional)
        comment: Order comment
        magic: Magic number
    
    Returns:
        dict with order result or error
    """
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    # Get current price
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        return {"success": False, "error": f"Could not get price for '{mt5_symbol}'"}
    
    # Use ask for buy, bid for sell
    price = tick.ask if order_type.upper() == "BUY" else tick.bid
    
    return place_limit_order(
        symbol=symbol,
        order_type=order_type,
        volume=volume,
        price=price,
        sl=sl,
        tp=tp,
        comment=comment,
        magic=magic,
    )


def execute_trade_signal(
    symbol: str,
    signal: str,  # "BUY", "SELL", "HOLD"
    entry_price: float = None,
    stop_loss: float = None,
    take_profit: float = None,
    volume: float = 0.01,
    use_limit_order: bool = True,
    comment: str = "TradingAgents",
) -> dict:
    """
    Execute a trade based on the analysis signal.
    
    This is the main function to call after report generation.
    
    Args:
        symbol: MT5 symbol (e.g., XAUUSD, XAGUSD)
        signal: Trade signal from analysis ("BUY", "SELL", "HOLD")
        entry_price: Entry price for limit order (uses current price if None)
        stop_loss: Stop loss price
        take_profit: Take profit price
        volume: Lot size (default 0.01 = micro lot)
        use_limit_order: If True, place limit order; if False, market order
        comment: Order comment
    
    Returns:
        dict with execution result
    """
    signal = signal.upper().strip()
    
    if signal == "HOLD" or signal not in ["BUY", "SELL"]:
        return {
            "success": True,
            "action": "HOLD",
            "message": f"No trade executed. Signal: {signal}",
        }
    
    _ensure_mt5_initialized()
    
    mt5_symbol = _resolve_symbol(symbol)
    
    # Get current price if entry_price not specified
    if entry_price is None:
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            return {"success": False, "error": f"Could not get price for '{mt5_symbol}'"}
        entry_price = tick.ask if signal == "BUY" else tick.bid
    
    # Determine order type
    if use_limit_order:
        order_type = "BUY_LIMIT" if signal == "BUY" else "SELL_LIMIT"
    else:
        order_type = signal
    
    # Place the order
    result = place_limit_order(
        symbol=mt5_symbol,
        order_type=order_type,
        volume=volume,
        price=entry_price,
        sl=stop_loss,
        tp=take_profit,
        comment=comment,
    )
    
    result["signal"] = signal
    result["entry_price"] = entry_price
    result["stop_loss"] = stop_loss
    result["take_profit"] = take_profit
    result["volume"] = volume
    
    return result


def get_open_positions(symbol: str = None) -> list:
    """Get all open positions, optionally filtered by symbol."""
    _ensure_mt5_initialized()
    
    if symbol:
        mt5_symbol = _resolve_symbol(symbol)
        positions = mt5.positions_get(symbol=mt5_symbol)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        return []
    
    return [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "comment": p.comment,
        }
        for p in positions
    ]


def get_pending_orders(symbol: str = None) -> list:
    """Get all pending orders, optionally filtered by symbol."""
    _ensure_mt5_initialized()
    
    if symbol:
        mt5_symbol = _resolve_symbol(symbol)
        orders = mt5.orders_get(symbol=mt5_symbol)
    else:
        orders = mt5.orders_get()
    
    if orders is None:
        return []
    
    order_types = {
        2: "BUY_LIMIT",
        3: "SELL_LIMIT",
        4: "BUY_STOP",
        5: "SELL_STOP",
    }
    
    return [
        {
            "ticket": o.ticket,
            "symbol": o.symbol,
            "type": order_types.get(o.type, str(o.type)),
            "volume": o.volume_current,
            "price": o.price_open,
            "sl": o.sl,
            "tp": o.tp,
            "comment": o.comment,
        }
        for o in orders
    ]


def cancel_order(ticket: int) -> dict:
    """Cancel a pending order by ticket number."""
    _ensure_mt5_initialized()
    
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        return {"success": False, "error": f"Cancel failed: {error}"}
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Cancel failed: {result.comment}"}
    
    return {"success": True, "ticket": ticket}


def modify_position(ticket: int, sl: float = None, tp: float = None) -> dict:
    """Modify stop loss and/or take profit of an open position."""
    _ensure_mt5_initialized()
    
    # Get position info
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return {"success": False, "error": f"Position {ticket} not found"}
    
    position = position[0]
    
    # Use existing values if not specified
    new_sl = sl if sl is not None else position.sl
    new_tp = tp if tp is not None else position.tp
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": new_tp,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        return {"success": False, "error": f"Modify failed: {error}"}
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Modify failed: {result.comment}", "retcode": result.retcode}
    
    return {
        "success": True,
        "ticket": ticket,
        "new_sl": new_sl,
        "new_tp": new_tp,
    }


def modify_order(ticket: int, price: float = None, sl: float = None, tp: float = None) -> dict:
    """Modify a pending order's price, stop loss, and/or take profit."""
    _ensure_mt5_initialized()
    
    # Get order info
    orders = mt5.orders_get(ticket=ticket)
    if not orders:
        return {"success": False, "error": f"Order {ticket} not found"}
    
    order = orders[0]
    
    # Use existing values if not specified
    new_price = price if price is not None else order.price_open
    new_sl = sl if sl is not None else order.sl
    new_tp = tp if tp is not None else order.tp
    
    request = {
        "action": mt5.TRADE_ACTION_MODIFY,
        "order": ticket,
        "price": new_price,
        "sl": new_sl,
        "tp": new_tp,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        return {"success": False, "error": f"Modify failed: {error}"}
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Modify failed: {result.comment}", "retcode": result.retcode}
    
    return {
        "success": True,
        "ticket": ticket,
        "new_price": new_price,
        "new_sl": new_sl,
        "new_tp": new_tp,
    }


def get_closed_deal_by_ticket(ticket: int, days_back: int = 30) -> dict:
    """
    Get closed deal info from MT5 history by position ticket.
    
    Args:
        ticket: The position ticket number
        days_back: How many days back to search in history
        
    Returns:
        dict with deal info including exit price, or None if not found
    """
    from datetime import timedelta
    
    _ensure_mt5_initialized()
    
    # Get deals from history
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now() + timedelta(days=1)
    
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None or len(deals) == 0:
        return None
    
    # Find deals matching this position ticket
    # Look for the closing deal (entry=1 means out/close)
    for deal in deals:
        if deal.position_id == ticket and deal.entry == 1:  # entry=1 means closing deal
            return {
                "ticket": deal.ticket,
                "position_id": deal.position_id,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == 0 else "SELL",
                "volume": deal.volume,
                "price": deal.price,  # This is the exit price
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "time": datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
            }
    
    return None


def get_history_deals(symbol: str = None, days_back: int = 30) -> list:
    """
    Get closed deals from MT5 history.
    
    Args:
        symbol: Filter by symbol (optional)
        days_back: How many days back to search
        
    Returns:
        List of closed deals
    """
    from datetime import timedelta
    
    _ensure_mt5_initialized()
    
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now() + timedelta(days=1)
    
    if symbol:
        mt5_symbol = _resolve_symbol(symbol)
        deals = mt5.history_deals_get(from_date, to_date, symbol=mt5_symbol)
    else:
        deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None:
        return []
    
    result = []
    for deal in deals:
        # entry: 0=in (open), 1=out (close), 2=reverse
        entry_type = {0: "OPEN", 1: "CLOSE", 2: "REVERSE"}.get(deal.entry, str(deal.entry))
        
        result.append({
            "ticket": deal.ticket,
            "position_id": deal.position_id,
            "symbol": deal.symbol,
            "type": "BUY" if deal.type == 0 else "SELL",
            "entry": entry_type,
            "volume": deal.volume,
            "price": deal.price,
            "profit": deal.profit,
            "commission": deal.commission,
            "swap": deal.swap,
            "time": datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
        })
    
    return result


def close_position(ticket: int) -> dict:
    """Close an open position by ticket number."""
    _ensure_mt5_initialized()
    
    # Get position info
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return {"success": False, "error": f"Position {ticket} not found"}
    
    position = position[0]
    
    # Determine close type (opposite of position type)
    close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    
    # Get current price
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return {"success": False, "error": "Could not get price"}
    
    price = tick.bid if position.type == 0 else tick.ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "TradingAgents close",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        return {"success": False, "error": f"Close failed: {error}"}
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Close failed: {result.comment}"}
    
    return {
        "success": True,
        "ticket": ticket,
        "profit": position.profit,
    }
