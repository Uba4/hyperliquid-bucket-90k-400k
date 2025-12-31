"""
Configuration for Hyperliquid Token Data Collection System
Bucket: 400k - 1.2M Volume Range
"""

# ===================== VOLUME BUCKET =====================
BUCKET_NAME = "400k-1.2m"
MIN_VOLUME = 400_000
MAX_VOLUME = 1_200_000

# ===================== API CONFIGURATION =====================
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
API_TIMEOUT = 10
API_DELAY = 0.1  # Delay between API calls (seconds)

# ===================== TECHNICAL ANALYSIS =====================
RSI_PERIOD = 14
MA_PERIOD = 14
SIGNAL_LINE = 50
LOOKBACK_PERIODS = 150  # Number of 15min bars to fetch for RSI calculation

# ===================== TRACKING CONFIGURATION =====================
TIMEFRAME = "15m"
POST_EXIT_TRACKING_DAYS = 10
BARS_PER_DAY = 96  # 24h * 4 bars/hour (15min bars)
DATA_RETENTION_DAYS = 365  # Keep data for 1 year

# ===================== FILE PATHS =====================
DATA_DIR = "data"
STATE_DIR = "state"
LOGS_DIR = "logs"
EXPORTS_DIR = "exports"

# State files
CURRENT_TOP6_FILE = f"{STATE_DIR}/current_top6.json"
ACTIVE_TRACKING_FILE = f"{STATE_DIR}/active_tracking.json"
TOKEN_METADATA_FILE = f"{STATE_DIR}/token_metadata.json"

# Log file
SYSTEM_LOG_FILE = f"{LOGS_DIR}/system.log"

# ===================== SYSTEM SETTINGS =====================
TOP_N = 6  # Track top 6 tokens

# ===================== DATA QUALITY =====================
MAX_MISSING_BARS_THRESHOLD = 5  # Alert if more than 5 bars are missing
VOLUME_RANGE_TOLERANCE = 0.1  # 10% tolerance for volume range checks
