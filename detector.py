"""
STAGE 1: TOP 6 DETECTOR
Runs every 5 minutes to detect Top 6 composition changes
"""

import time
from datetime import datetime
from typing import List, Tuple, Dict
import config
import utils

def calculate_top6_scores(tokens: List[str], data_cache: Dict) -> List[Tuple[str, int]]:
    """
    Calculate scores for all tokens via pairwise ratio analysis
    Returns: List of (token, score) tuples sorted by score descending
    """
    scores = {token: 0 for token in tokens}
    
    total_comparisons = len(tokens) * (len(tokens) - 1)
    successful_comparisons = 0
    
    utils.logger.info(f"🔄 Analyzing {total_comparisons:,} ratio pairs...")
    
    for token1 in tokens:
        for token2 in tokens:
            if token1 == token2:
                continue
            
            winner, loser = utils.analyze_ratio(token1, token2, data_cache)
            if winner and loser:
                scores[winner] += 1
                successful_comparisons += 1
            
            time.sleep(config.API_DELAY)
    
    success_rate = (successful_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0
    utils.logger.info(f"✅ Completed: {successful_comparisons:,}/{total_comparisons:,} successful ({success_rate:.1f}%)")
    
    # Sort by score descending and return
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def detect_changes(previous_top6: Dict, current_top6: List[str]) -> Dict:
    """
    Detect changes in Top 6 composition
    Returns: dict with 'new_entries' and 'dropped_out' lists
    """
    if not previous_top6 or 'tokens' not in previous_top6:
        return {'new_entries': current_top6, 'dropped_out': []}
    
    prev_tokens = set(previous_top6['tokens'])
    curr_tokens = set(current_top6)
    
    new_entries = sorted(curr_tokens - prev_tokens)
    dropped_out = sorted(prev_tokens - curr_tokens)
    
    return {
        'new_entries': new_entries,
        'dropped_out': dropped_out
    }

def update_active_tracking(changes: Dict, active_tracking: Dict, volume_data: Dict):
    """
    Update active tracking based on Top 6 changes
    
    Active tracking includes:
    - Tokens currently in Top 6
    - Tokens in post-exit tracking window (up to 10 days)
    """
    now = datetime.utcnow()
    
    # Process new entries
    for token in changes['new_entries']:
        if token not in active_tracking:
            # Brand new token
            active_tracking[token] = {
                'current_run': 1,
                'status': 'in_top6',
                'entry_time': now.isoformat(),
                'exit_time': None,
                'last_bar_collected': 0,
                'tracking_until': None,
                'entry_volume': volume_data.get(token, 0)
            }
            utils.logger.info(f"🆕 New tracking: {token} (Run 1)")
        else:
            # Re-entry during post-exit window
            existing = active_tracking[token]
            if existing['status'] == 'post_exit':
                # Start new run
                existing['current_run'] += 1
                existing['status'] = 'in_top6'
                existing['entry_time'] = now.isoformat()
                existing['exit_time'] = None
                existing['tracking_until'] = None
                existing['entry_volume'] = volume_data.get(token, 0)
                utils.logger.info(f"🔄 Re-entry: {token} (Run {existing['current_run']})")
    
    # Process dropouts
    for token in changes['dropped_out']:
        if token in active_tracking:
            existing = active_tracking[token]
            existing['status'] = 'post_exit'
            existing['exit_time'] = now.isoformat()
            # Set tracking_until to 10 days from now
            tracking_until = now + timedelta(days=config.POST_EXIT_TRACKING_DAYS)
            existing['tracking_until'] = tracking_until.isoformat()
            utils.logger.info(f"📉 Dropout: {token} (tracking until {utils.format_timestamp(tracking_until)})")
    
    # Clean up expired tracking
    tokens_to_remove = []
    for token, data in active_tracking.items():
        if data['status'] == 'post_exit' and data['tracking_until']:
            tracking_until = datetime.fromisoformat(data['tracking_until'])
            if now > tracking_until:
                tokens_to_remove.append(token)
                utils.logger.info(f"✅ Completed tracking: {token} (Run {data['current_run']})")
    
    for token in tokens_to_remove:
        del active_tracking[token]
    
    return active_tracking

def main():
    """Main detector logic"""
    utils.initialize_system()
    
    utils.logger.info("=" * 70)
    utils.logger.info(f"🔍 DETECTOR START - {utils.format_timestamp(datetime.utcnow())}")
    utils.logger.info("=" * 70)
    
    # Load existing state
    previous_top6 = utils.load_json(config.CURRENT_TOP6_FILE)
    active_tracking = utils.load_json(config.ACTIVE_TRACKING_FILE, default={})
    
    # Step 1: Fetch all tokens and volumes
    utils.logger.info("\n📡 Fetching market data...")
    all_tokens = utils.fetch_perpetuals_meta()
    if not all_tokens:
        utils.logger.error("❌ Failed to fetch tokens. Aborting.")
        return
    
    volume_data = utils.fetch_asset_contexts()
    if not volume_data:
        utils.logger.error("❌ Failed to fetch volumes. Aborting.")
        return
    
    # Step 2: Filter tokens by volume range
    filtered_tokens = [
        token for token in all_tokens 
        if utils.is_in_volume_range(volume_data.get(token, 0))
    ]
    
    utils.logger.info(f"✅ {len(filtered_tokens)} tokens in ${config.MIN_VOLUME:,} - ${config.MAX_VOLUME:,} range")
    
    if len(filtered_tokens) < config.TOP_N:
        utils.logger.error(f"❌ Not enough tokens ({len(filtered_tokens)} < {config.TOP_N}). Aborting.")
        return
    
    # Step 3: Fetch 15min candles for ratio analysis
    utils.logger.info(f"\n📊 Fetching {config.TIMEFRAME} candles for {len(filtered_tokens)} tokens...")
    data_cache = {}
    
    for i, token in enumerate(filtered_tokens):
        df = utils.fetch_candles(token, config.TIMEFRAME, config.LOOKBACK_PERIODS)
        if not df.empty:
            data_cache[token] = df
        
        if (i + 1) % 10 == 0:
            utils.logger.info(f"   Progress: {i + 1}/{len(filtered_tokens)}")
        
        time.sleep(config.API_DELAY)
    
    utils.logger.info(f"✅ Fetched data for {len(data_cache)} tokens")
    
    if len(data_cache) < config.TOP_N:
        utils.logger.error(f"❌ Insufficient data ({len(data_cache)} < {config.TOP_N}). Aborting.")
        return
    
    # Step 4: Calculate Top 6
    utils.logger.info("\n🧮 Calculating Top 6 rankings...")
    sorted_scores = calculate_top6_scores(list(data_cache.keys()), data_cache)
    top6_with_scores = sorted_scores[:config.TOP_N]
    top6_tokens = [token for token, score in top6_with_scores]
    
    # Step 5: Display results
    utils.logger.info("\n" + "=" * 70)
    utils.logger.info(f"⚡ TOP {config.TOP_N} - {datetime.utcnow().strftime('%H:%M UTC')}")
    utils.logger.info("=" * 70)
    for rank, (token, score) in enumerate(top6_with_scores, 1):
        vol = volume_data.get(token, 0)
        utils.logger.info(f"#{rank}  {token:8s}  Score: {score:3d}  Volume: {utils.format_volume(vol)}")
    utils.logger.info("=" * 70)
    
    # Step 6: Detect changes
    changes = detect_changes(previous_top6, top6_tokens)
    
    if changes['new_entries'] or changes['dropped_out']:
        utils.logger.info("\n📢 COMPOSITION CHANGES DETECTED")
        if changes['new_entries']:
            utils.logger.info(f"   ➕ New entries: {', '.join(changes['new_entries'])}")
        if changes['dropped_out']:
            utils.logger.info(f"   ➖ Dropped out: {', '.join(changes['dropped_out'])}")
    else:
        utils.logger.info("\n✅ No composition changes")
    
    # Step 7: Update active tracking
    active_tracking = update_active_tracking(changes, active_tracking, volume_data)
    
    # Step 8: Save state
    current_top6_state = {
        'tokens': top6_tokens,
        'scores': [score for token, score in top6_with_scores],
        'timestamp': datetime.utcnow().isoformat(),
        'bucket': config.BUCKET_NAME
    }
    
    utils.save_json(config.CURRENT_TOP6_FILE, current_top6_state)
    utils.save_json(config.ACTIVE_TRACKING_FILE, active_tracking)
    
    # Step 9: Summary
    in_top6_count = sum(1 for data in active_tracking.values() if data['status'] == 'in_top6')
    post_exit_count = sum(1 for data in active_tracking.values() if data['status'] == 'post_exit')
    
    utils.logger.info(f"\n📊 Active Tracking Summary:")
    utils.logger.info(f"   • In Top 6: {in_top6_count} tokens")
    utils.logger.info(f"   • Post-exit: {post_exit_count} tokens")
    utils.logger.info(f"   • Total active: {len(active_tracking)} tokens")
    
    utils.logger.info("\n" + "=" * 70)
    utils.logger.info(f"✅ DETECTOR COMPLETE - {utils.format_timestamp(datetime.utcnow())}")
    utils.logger.info("=" * 70)

if __name__ == "__main__":
    try:
        from datetime import timedelta  # Import here for active_tracking update
        main()
    except Exception as e:
        utils.logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        raise
