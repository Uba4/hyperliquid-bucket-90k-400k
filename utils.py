"""
Shared utility functions for Hyperliquid data collection system
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

import config

# ===================== LOGGING SETUP =====================

def setup_logging():
    """Initialize logging system"""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(config.SYSTEM_LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ===================== DIRECTORY MANAGEMENT =====================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [config.DATA_DIR, config.STATE_DIR, config.LOGS_DIR, config.EXPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    logger.info("✅ Directories initialized")

# ===================== FILE I/O =====================

def load_json(filepath: str, default: dict = None) -> dict:
    """Load JSON file with error handling"""
    if default is None:
        default = {}
    
    if not os.path.exists(filepath):
        return default
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")
        return default

def save_json(filepath: str, data: dict):
    """Save JSON file with error handling"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Error saving {filepath}: {e}")

def load_parquet(filepath: str) -> Optional[pd.DataFrame]:
    """Load Parquet file with error handling"""
    if not os.path.exists(filepath):
        return None
    
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")
        return None

def save_parquet(filepath: str, df: pd.DataFrame):
    """Save Parquet file with error handling"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False, compression='snappy')
    except Exception as e:
        logger.error(f"❌ Error saving {filepath}: {e}")

# ===================== API FUNCTIONS =====================

def fetch_perpetuals_meta() -> List[str]:
    """Fetch all available perpetual tokens from Hyperliquid"""
    payload = {"type": "meta"}
    try:
        response = requests.post(config.HYPERLIQUID_API, json=payload, timeout=config.API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        tokens = [asset['name'] for asset in data['universe']]
        logger.info(f"✅ Fetched {len(tokens)} perpetual tokens")
        return tokens
    except Exception as e:
        logger.error(f"❌ Error fetching perpetuals: {e}")
        return []

def fetch_asset_contexts() -> Dict[str, float]:
    """Fetch 24h volume for all tokens"""
    payload = {"type": "metaAndAssetCtxs"}
    try:
        response = requests.post(config.HYPERLIQUID_API, json=payload, timeout=config.API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        volume_data = {}
        for i, ctx in enumerate(data[1]):
            token_name = data[0]['universe'][i]['name']
            volume_24h = float(ctx.get('dayNtlVlm', 0))
            volume_data[token_name] = volume_24h
        
        logger.info(f"✅ Fetched volume data for {len(volume_data)} tokens")
        return volume_data
    except Exception as e:
        logger.error(f"❌ Error fetching volumes: {e}")
        return {}

def fetch_candles(token: str, interval: str = "15m", periods: int = None) -> pd.DataFrame:
    """Fetch OHLCV candles for a token"""
    if periods is None:
        periods = config.LOOKBACK_PERIODS
    
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = interval_minutes.get(interval, 15)
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (periods * minutes * 60 * 1000)
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": token,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time
        }
    }
    
    try:
        response = requests.post(config.HYPERLIQUID_API, json=payload, timeout=config.API_TIMEOUT)
        response.raise_for_status()
        candles = response.json()
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['close'] = df['c'].astype(float)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[['timestamp', 'close']]
    except Exception as e:
        logger.warning(f"⚠️  Error fetching candles for {token}: {e}")
        return pd.DataFrame()

def fetch_candles_range(token: str, start_time: datetime, end_time: datetime, 
                       interval: str = "15m") -> pd.DataFrame:
    """Fetch candles for a specific time range"""
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = interval_minutes.get(interval, 15)
    
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": token,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms
        }
    }
    
    try:
        response = requests.post(config.HYPERLIQUID_API, json=payload, timeout=config.API_TIMEOUT)
        response.raise_for_status()
        candles = response.json()
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df['close'] = df['c'].astype(float)
        df['volume'] = df['v'].astype(float)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[['timestamp', 'close', 'volume']]
    except Exception as e:
        logger.warning(f"⚠️  Error fetching candles for {token}: {e}")
        return pd.DataFrame()

# ===================== TECHNICAL ANALYSIS =====================

def calculate_rsi(prices: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi_ma(rsi: pd.Series, period: int = config.MA_PERIOD) -> float:
    """Calculate RSI moving average"""
    rsi_ma = rsi.rolling(window=period).mean()
    return rsi_ma.iloc[-1] if len(rsi_ma) > 0 and not pd.isna(rsi_ma.iloc[-1]) else 50.0

def create_ratio_series(token1_data: pd.DataFrame, token2_data: pd.DataFrame) -> pd.Series:
    """Create ratio between two token price series"""
    merged = pd.merge(token1_data, token2_data, on='timestamp', suffixes=('_1', '_2'))
    if len(merged) == 0:
        return pd.Series()
    return merged['close_1'] / merged['close_2']

def analyze_ratio(token1: str, token2: str, data_cache: Dict[str, pd.DataFrame]) -> Tuple[Optional[str], Optional[str]]:
    """
    Analyze ratio between two tokens using RSI-MA
    Returns: (winner, loser) or (None, None) if analysis fails
    """
    if token1 not in data_cache or token2 not in data_cache:
        return None, None
    
    if data_cache[token1].empty or data_cache[token2].empty:
        return None, None
    
    ratio = create_ratio_series(data_cache[token1], data_cache[token2])
    
    if len(ratio) < config.RSI_PERIOD + config.MA_PERIOD:
        return None, None
    
    rsi = calculate_rsi(ratio, config.RSI_PERIOD)
    rsi_ma = calculate_rsi_ma(rsi, config.MA_PERIOD)
    
    if rsi_ma > config.SIGNAL_LINE:
        return token1, token2
    else:
        return token2, token1

# ===================== FORMATTING UTILITIES =====================

def format_volume(volume: float) -> str:
    """Format volume for display"""
    if volume >= 1_000_000:
        return f"${volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.1f}K"
    else:
        return f"${volume:.0f}"

def format_timestamp(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime('%Y-%m-%d %H:%M UTC')

def format_price(price: float) -> str:
    """Format price for display"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

# ===================== DATA VALIDATION =====================

def is_in_volume_range(volume: float) -> bool:
    """Check if volume is within bucket range (with tolerance)"""
    tolerance = config.VOLUME_RANGE_TOLERANCE
    min_vol = config.MIN_VOLUME * (1 - tolerance)
    max_vol = config.MAX_VOLUME * (1 + tolerance)
    return min_vol < volume <= max_vol

def validate_bar_continuity(df: pd.DataFrame) -> Tuple[bool, List[int]]:
    """
    Check if bars are continuous (no gaps)
    Returns: (is_continuous, list_of_missing_bar_numbers)
    """
    if df.empty:
        return True, []
    
    expected_bars = set(range(1, df['bar'].max() + 1))
    actual_bars = set(df['bar'].values)
    missing_bars = sorted(expected_bars - actual_bars)
    
    return len(missing_bars) == 0, missing_bars

# ===================== EXPORT UTILITIES =====================

def export_token_to_csv(token: str) -> bool:
    """Export a token's Parquet data to CSV for manual inspection"""
    parquet_path = f"{config.DATA_DIR}/{token}.parquet"
    csv_path = f"{config.EXPORTS_DIR}/{token}.csv"
    
    df = load_parquet(parquet_path)
    if df is None:
        logger.error(f"❌ Cannot export {token}: Parquet file not found")
        return False
    
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Exported {token} to CSV ({len(df)} rows)")
        return True
    except Exception as e:
        logger.error(f"❌ Error exporting {token} to CSV: {e}")
        return False

# ===================== TIME UTILITIES =====================

def get_bar_count_for_days(days: int) -> int:
    """Calculate number of 15min bars for given days"""
    return days * config.BARS_PER_DAY

def get_days_from_bars(bars: int) -> float:
    """Calculate days from number of 15min bars"""
    return bars / config.BARS_PER_DAY

def round_to_15min(dt: datetime) -> datetime:
    """Round datetime to nearest 15-minute mark"""
    minutes = (dt.minute // 15) * 15
    return dt.replace(minute=minutes, second=0, microsecond=0)

# ===================== INITIALIZATION =====================

def initialize_system():
    """Initialize the system on first run"""
    ensure_directories()
    logger.info("=" * 70)
    logger.info(f"🚀 Hyperliquid Data Collection System - Bucket {config.BUCKET_NAME}")
    logger.info("=" * 70)
    logger.info(f"Volume Range: ${config.MIN_VOLUME:,} - ${config.MAX_VOLUME:,}")
    logger.info(f"Timeframe: {config.TIMEFRAME}")
    logger.info(f"Post-exit tracking: {config.POST_EXIT_TRACKING_DAYS} days")
    logger.info(f"Data retention: {config.DATA_RETENTION_DAYS} days")
    logger.info("=" * 70)
