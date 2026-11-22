import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("NOISE_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"
REFERENCE = DATA / "reference"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

BINANCE_BULK = "https://data.binance.vision"
BINANCE_FAPI = "https://fapi.binance.com"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "")
COINGECKO_PLAN = os.environ.get("COINGECKO_PLAN", "pro")
COINGECKO_PAUSE = 0.15 if COINGECKO_PLAN == "pro" else 2.1

DATA_START = "2024-08-01"
DATA_END = "2026-11-01"
FIRST_UNIVERSE_MONTH = "2024-11"

LOOKBACK_DAYS = 30
MIN_LISTED_DAYS = 365
MIN_TRADED_SHARE = 0.90
MIN_VOLUME_USD = 3_000_000
NOT_COINS = ["BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL", "ALL"]
STABLECOINS = [
    "USDC",
    "FDUSD",
    "TUSD",
    "USDP",
    "DAI",
    "BUSD",
    "USDE",
    "PYUSD",
    "EUR",
    "USD1",
    "RLUSD",
    "XUSD",
]
STABLE_WORDS = ["stablecoin"]

SPIKE_LOG = 0.3
SPIKE_WINDOW_HOURS = 49
MAX_FILL_HOURS = 24
MIN_COVERAGE = 0.95

REPICK = os.environ.get("NOISE_REPICK", "M")

FORMATION_DAYS = 90
RUN = f"{'weekly' if REPICK == 'W' else 'monthly'}_{FORMATION_DAYS}d" + os.environ.get("NOISE_RUN_TAG", "")
RUN_DIR = PROCESSED / RUN
MIN_FORMATION_COVERAGE = 0.95
MIN_CORR_LOGP = 0.9
MIN_RESID_CORR = 0.2

MAX_CATEGORY_SIZE = 40

NOT_A_REASON = [
    "portfolio",
    "index",
    "made in",
    "launchpool",
    "launchpad",
    "holdings",
    "alleged",
    "airdrop",
    "ventures",
    "capital",
    "labs",
    "binance alpha",
    "listing",
    "coinlist",
    "ecosystem",
]
LOOKALIKES = 5

MAX_PAIRS, MAX_BASKETS, MAX_SECTOR = 300, 100, 100
SEED = 0

JOHANSEN_LAGS = 1
SIGNIFICANCE = 0.05

FEE_PER_SIDE = 0.0005
SLIPPAGE_BP = [(100e6, 2), (20e6, 5), (0, 10)]

MIN_HALF_LIFE_H = 6
MAX_HALF_LIFE_H = 240
MIN_SWING_TO_COST = 3

YARDSTICK_ENTRY_Z = 2

TOP_N = 20

BT_STRATEGY = "band"

BT_ENTRY_Z = 2.0
BT_EXIT_Z = 0.0
BT_STOP_Z = 4.0
BT_FULL_Z = 2.0
BT_STEP = 0.25
BT_CAPITAL = 1_000_000
BT_LEVERAGE = 2.0
BT_MIN_SPREADS = 20

BT_MAX_ADV = 0.02
BT_IMPACT_Y = 0.5
BT_HOLDOUT_FROM = "2026-05"

KF_WARMUP_DAYS = 30
KF_MEMORY_DAYS = 30
KF_BAND_DAYS = 10
KF_ENTRY = 3.0
KF_STOP = 5.0
