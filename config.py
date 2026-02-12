import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_KEY = os.getenv('MEXC_API_KEY', '')
API_SECRET = os.getenv('MEXC_API_SECRET', '')
TESTNET_ENABLED = os.getenv('TESTNET_ENABLED', 'False').lower() == 'true'
MEXC_API_URL = 'https://testnet.mexc.com' if TESTNET_ENABLED else 'https://api.mexc.com'

# Trading Parameters - Defaults
DEFAULT_MIN_SPREAD = 0.5  # %
DEFAULT_TRADE_VOLUME = 100  # USDT
DEFAULT_LEVERAGE = 5  # 1-125x
DEFAULT_TAKE_PROFIT = 2.0  # %
DEFAULT_STOP_LOSS = 1.0  # %
DEFAULT_COINS_TO_SCAN = 50  # 10-500
DEFAULT_SCAN_INTERVAL = 10  # seconds (1-30)
DEFAULT_MAX_POSITIONS = 3  # 1-10

# Risk Management
MAX_DAILY_LOSS_USDT = 500  # Stop trading if daily loss exceeds this
COIN_COOLDOWN_SECONDS = 300  # Don't trade same coin within this period
MIN_DEX_LIQUIDITY = 10000  # USDT minimum liquidity on DEX
ANOMALY_SPREAD_THRESHOLD = 20  # % - Alert if spread exceeds this

# API Rate Limits
MEXC_RATE_LIMIT = 10  # requests per second
DEXSCREENER_RATE_LIMIT = 5  # requests per second
COINGECKO_RATE_LIMIT = 10  # requests per second

# Data Storage
DATA_DIR = Path('data')
LOGS_DIR = Path('logs')
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# UI Configuration
DARK_THEME = True
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
CHART_UPDATE_INTERVAL = 1000  # milliseconds

# Supported Exchanges for DEX
SUPPORTED_DEXES = ['uniswap', 'pancakeswap', 'traderjoe', 'sushiswap']

# Stablecoin detection
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP']