"""Central configuration for the Olist customer-analysis portfolio project.

All tunable parameters live here so every script stays reproducible and each
decision is traceable. See docs/project-plan.md and logs/decisions.log.
"""
from pathlib import Path
from datetime import datetime, timedelta

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
OUTPUTS_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "olist.db"

# --- Temporal design (decided 2026-06-15; see logs/decisions.log) ---
# Snapshot date = latest order_purchase_timestamp in the dataset (verified).
# Acts as the reference "today" for RFM Recency.
SNAPSHOT_DATE = datetime(2018, 10, 17)

# Module 4 (repurchase propensity / high-potential-customer mining):
# label = the customer placed a 2nd order within this window after their 1st order.
REPURCHASE_WINDOW_DAYS = 180
# Only customers whose FIRST order is on/before this cutoff have a full
# observable window -> prevents look-ahead leakage. Later first-orders are
# censored and excluded from the modelling set.
MODEL_FIRST_ORDER_CUTOFF = SNAPSHOT_DATE - timedelta(days=REPURCHASE_WINDOW_DAYS)  # 2018-04-20

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Module 3 (RFM K-means) ---
# k is chosen in stage 3 by silhouette + business interpretability.
# These are CANDIDATES to scan, NOT a hardcoded final choice.
KMEANS_K_CANDIDATES = range(2, 9)
