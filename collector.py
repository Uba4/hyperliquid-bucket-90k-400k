"""
STAGE 2: DATA COLLECTOR
Runs every 60 minutes to collect 15min bar data for all active tokens
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import config
import utils

def initialize_token_parquet(token: str, entry_time: datetime, run: int) -> pd.DataFrame:
    """
    Initialize a new Parquet file for a token or load existing one
    """
    filepath = f"{config.DATA_DIR}/{token}.parquet"
    
    # Try to load existing file
    existing_df = utils.load_parquet(filepath)
    
    if existing_df is not None:
        utils.logger.info(f"📂 Loaded existing data for {token}: {len(existing_df)} bars")
        return existing_df
    
    # Create new empty dataframe with correct schema
    df = pd.DataFrame(columns=['bar', 'timestamp', 'run', 'in_top6', 'price', 'volume_24h', 'btc_price'])
    df = df.astype({
        'bar': 'int64',
        'timestamp': 'datetime64[ns]',
        'run': 'int64',
        'in_top6': 'bool',
        'price': 'float64',
        'volume_24h': 'float64',
        'btc_price': 'float64'
    })
    
    utils.logger.info(f"🆕 Initialized new Parquet for {token}")
    return df

def get_next_bar_timestamp(df: pd.DataFrame, entry_time: datetime) -> datetime:
    """
    Determine the next timestamp we need to collect
    """
    if df.empty:
        # Start from entry time, rounded to nearest 15min
        return utils.round_to_15min(entry_time)
    
    # Get last timestamp and add 15 minutes
    last_timestamp = df['timestamp'].max()
    return last_timestamp + timedelta(minutes=15)

def calculate_24h_volume(token: str, timestamp: datetime, volume_data: Dict[str, float]) -> float:
    """
    Calculate rolling 24h volume at given timestamp
    
    Note: For simplicity, we use the current 24h volume from API.
    For true historical 24h volume, we'd need to sum the last 96 bars.
    """
    return volume_data.get(token, 0)

def collect_bars_for_token(token: str, tracking_data: Dict, volume_data: Dict[str, float], 
                           btc_data: pd.DataFrame) -> bool:
    """
    Collect all missing bars for a token since last collection
    
    Returns: True if successful, False otherwise
    """
    # Load or initialize Parquet file
    df = initialize_token_parquet(
        token, 
        datetime.fromisoformat(tracking_data['entry_time']),
        tracking_data['current_run']
    )
    
    # Determine what we need to collect
    entry_time = datetime.fromisoformat(tracking_data['entry_time'])
    current_time = datetime.utcnow()
    current_run = tracking_data['current_run']
    is_in_top6 = tracking_data['status'] == 'in_top6'
    
    # Find the timestamp we need to start from
    if df.empty or df[df['run'] == current_run].empty:
        # New run or empty file - start from entry time
        next_timestamp = utils.round_to_15min(entry_time)
    else:
        # Continue from last collected bar in this run
        run_df = df[df['run'] == current_run]
        last_timestamp = run_df['timestamp'].max()
        next_timestamp = last_timestamp + timedelta(minutes=15)
    
    # Calculate how many bars to collect
    time_diff = current_time - next_timestamp
    bars_to_collect = int(time_diff.total_seconds() / 900)  # 900 seconds = 15 min
    
    if bars_to_collect <= 0:
        utils.logger.info(f"⏭️  {token}: No new bars to collect")
        return True
    
    # Limit to prevent excessive API calls in case of long gaps
    if bars_to_collect > 100:
        utils.logger.warning(f"⚠️  {token}: Large gap detected ({bars_to_collect} bars). Collecting last 100.")
        bars_to_collect = 100
        next_timestamp = current_time - timedelta(minutes=15 * 100)
    
    # Fetch candle data for the time range
    utils.logger.info(f"📥 {token}: Fetching {bars_to_collect} bars from {utils.format_timestamp(next_timestamp)}")
    
    end_time = current_time
    candles = utils.fetch_candles_range(token, next_timestamp, end_time, config.TIMEFRAME)
    
    if candles.empty:
        utils.logger.warning(f"⚠️  {token}: No candle data returned")
        return False
    
    # Merge with BTC data
    candles = pd.merge(candles, btc_data, on='timestamp', how='left', suffixes=('', '_btc'))
    candles['btc_price'] = candles['close_btc']
    
    # Get current 24h volume
    current_volume_24h = calculate_24h_volume(token, current_time, volume_data)
    
    # Create new rows
    new_rows = []
    start_bar = df['bar'].max() + 1 if not df.empty else 1
    
    for i, row in candles.iterrows():
        bar_num = start_bar + i
        new_row = {
            'bar': bar_num,
            'timestamp': row['timestamp'],
            'run': current_run,
            'in_top6': is_in_top6,
            'price': row['close'],
            'volume_24h': current_volume_24h,
            'btc_price': row['btc_price'] if pd.notna(row['btc_price']) else None
        }
        new_rows.append(new_row)
    
    # Append new data
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Validate continuity
    is_continuous, missing_bars = utils.validate_bar_continuity(combined_df)
    if not is_continuous:
        utils.logger.warning(f"⚠️  {token}: Missing bars detected: {missing_bars}")
    
    # Save updated Parquet
    filepath = f"{config.DATA_DIR}/{token}.parquet"
    utils.save_parquet(filepath, combined_df)
    
    # Update tracking
    tracking_data['last_bar_collected'] = int(combined_df['bar'].max())
    
    utils.logger.info(f"✅ {token}: Collected {len(new_rows)} bars (total: {len(combined_df)} bars)")
    
    return True

def clean_old_data():
    """
    Remove data older than retention period (1 year)
    
    Note: Currently a placeholder. Can be implemented to archive/delete old Parquet files.
    """
    # Implementation would check token_metadata for tokens that exited > 365 days ago
    pass

def main():
    """Main collector logic"""
    utils.initialize_system()
    
    utils.logger.info("=" * 70)
    utils.logger.info(f"📦 COLLECTOR START - {utils.format_timestamp(datetime.utcnow())}")
    utils.logger.info("=" * 70)
    
    # Load state
    active_tracking = utils.load_json(config.ACTIVE_TRACKING_FILE, default={})
    
    if not active_tracking:
        utils.logger.info("ℹ️  No active tokens to track. Exiting.")
        return
    
    utils.logger.info(f"📊 Active tracking: {len(active_tracking)} tokens")
    
    # Fetch current market data (for volume)
    utils.logger.info("\n📡 Fetching market data...")
    volume_data = utils.fetch_asset_contexts()
    
    if not volume_data:
        utils.logger.error("❌ Failed to fetch volume data. Aborting.")
        return
    
    # Fetch BTC data (common reference for all tokens)
    utils.logger.info("📊 Fetching BTC reference data...")
    btc_candles = utils.fetch_candles('BTC', config.TIMEFRAME, periods=200)
    
    if btc_candles.empty:
        utils.logger.warning("⚠️  Failed to fetch BTC data. Continuing without BTC reference.")
        btc_candles = pd.DataFrame(columns=['timestamp', 'close'])
        btc_candles.rename(columns={'close': 'close_btc'}, inplace=True)
    else:
        btc_candles.rename(columns={'close': 'close_btc'}, inplace=True)
    
    # Process each active token
    utils.logger.info("\n🔄 Collecting data for active tokens...")
    
    success_count = 0
    failure_count = 0
    
    for i, (token, tracking_data) in enumerate(active_tracking.items(), 1):
        utils.logger.info(f"\n[{i}/{len(active_tracking)}] Processing {token} (Run {tracking_data['current_run']}, {tracking_data['status']})")
        
        try:
            success = collect_bars_for_token(token, tracking_data, volume_data, btc_candles)
            if success:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            utils.logger.error(f"❌ Error collecting {token}: {e}")
            failure_count += 1
        
        # Rate limiting
        time.sleep(config.API_DELAY)
    
    # Save updated tracking state
    utils.save_json(config.ACTIVE_TRACKING_FILE, active_tracking)
    
    # Summary
    utils.logger.info("\n" + "=" * 70)
    utils.logger.info("📊 COLLECTION SUMMARY")
    utils.logger.info("=" * 70)
    utils.logger.info(f"✅ Successful: {success_count}/{len(active_tracking)}")
    if failure_count > 0:
        utils.logger.info(f"❌ Failed: {failure_count}/{len(active_tracking)}")
    
    utils.logger.info("\n" + "=" * 70)
    utils.logger.info(f"✅ COLLECTOR COMPLETE - {utils.format_timestamp(datetime.utcnow())}")
    utils.logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        utils.logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        raise
