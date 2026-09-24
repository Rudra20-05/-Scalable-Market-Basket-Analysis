"""
config.py
---------
Central configuration file for the Scalable Market Basket Analysis project.

Every path, default parameter, and constant used across the project is
defined here so that changing a setting in one place updates the whole
pipeline. Keeping configuration centralized also makes the project easier
to grade/understand for academic submission purposes.
"""

import os

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# BASE_DIR points to the project's root folder (one level above /config)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# DIRECTORY PATHS
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
RAW_DATASET_PATH = os.path.join(DATA_DIR, "retail_transactions.csv")

FREQUENT_ITEMSETS_PATH = os.path.join(RESULTS_DIR, "frequent_itemsets.csv")
ASSOCIATION_RULES_PATH = os.path.join(RESULTS_DIR, "association_rules.csv")
BUSINESS_INSIGHTS_PATH = os.path.join(RESULTS_DIR, "business_insights.csv")
PERFORMANCE_RESULTS_PATH = os.path.join(RESULTS_DIR, "performance_results.csv")
PERFORMANCE_PLOT_PATH = os.path.join(RESULTS_DIR, "performance_analysis.png")
EVALUATION_SUMMARY_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.json")
PIPELINE_LOG_PATH = os.path.join(RESULTS_DIR, "pipeline_run_log.txt")

# ---------------------------------------------------------------------------
# DATASET GENERATION DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_NUM_TRANSACTIONS = 10000
RANDOM_SEED = 42

# Dataset sizes used by the performance / scalability test script.
# NOTE: These can be edited freely. Larger sizes take longer to run,
# especially the FP-Growth stage and Spark session startup.
PERFORMANCE_TEST_SIZES = [10000, 50000, 100000, 500000]

# ---------------------------------------------------------------------------
# PRODUCT CATALOG
# ---------------------------------------------------------------------------
# category -> list of (product_name, base_price)
PRODUCT_CATALOG = {
    "Grocery": [
        ("Bread", 40), ("Rice", 60), ("Cooking Oil", 120), ("Pasta", 55),
        ("Pasta Sauce", 70), ("Cereal", 150), ("Biscuits", 30), ("Sugar", 45),
    ],
    "Dairy": [
        ("Milk", 28), ("Butter", 55), ("Cheese", 90), ("Eggs", 65),
        ("Yogurt", 35),
    ],
    "Beverages": [
        ("Coffee", 180), ("Tea", 110), ("Juice", 60), ("Soft Drink", 40),
    ],
    "Snacks": [
        ("Chips", 25), ("Chocolate", 50), ("Namkeen", 35), ("Cookies", 45),
    ],
    "Personal Care": [
        ("Shampoo", 140), ("Soap", 35), ("Toothpaste", 65),
        ("Toothbrush", 30),
    ],
    "Household": [
        ("Dish Soap", 55), ("Detergent", 130), ("Tissue Paper", 45),
        ("Garbage Bags", 60),
    ],
}

# Flattened list of all product names (used for validation / lookups)
ALL_PRODUCTS = [name for items in PRODUCT_CATALOG.values() for name, _ in items]

# Product -> category lookup, built at import time
PRODUCT_TO_CATEGORY = {
    name: category
    for category, items in PRODUCT_CATALOG.items()
    for name, _ in items
}

# Product -> base price lookup
PRODUCT_TO_PRICE = {
    name: price
    for items in PRODUCT_CATALOG.values()
    for name, price in items
}

# ---------------------------------------------------------------------------
# REALISTIC ASSOCIATION RULES USED ONLY FOR *GENERATING* THE SYNTHETIC DATA
# ---------------------------------------------------------------------------
# These describe the probability that a "companion" product is added to a
# basket when a "trigger" product is present. FP-Growth is never told about
# this dictionary — it must rediscover these relationships purely from the
# generated transactions. This is what makes the synthetic data realistic
# instead of purely random.
ASSOCIATION_SEEDS = [
    ("Bread", "Butter", 0.55),
    ("Bread", "Eggs", 0.35),
    ("Milk", "Cereal", 0.50),
    ("Milk", "Bread", 0.30),
    ("Pasta", "Pasta Sauce", 0.65),
    ("Pasta", "Cheese", 0.30),
    ("Coffee", "Sugar", 0.45),
    ("Coffee", "Milk", 0.30),
    ("Tea", "Sugar", 0.40),
    ("Chips", "Soft Drink", 0.50),
    ("Chocolate", "Soft Drink", 0.25),
    ("Shampoo", "Soap", 0.30),
    ("Toothpaste", "Toothbrush", 0.45),
    ("Dish Soap", "Detergent", 0.25),
    ("Rice", "Cooking Oil", 0.35),
    ("Cookies", "Tea", 0.30),
]

# ---------------------------------------------------------------------------
# FP-GROWTH DEFAULT PARAMETERS
# ---------------------------------------------------------------------------
DEFAULT_MIN_SUPPORT = 0.01
DEFAULT_MIN_CONFIDENCE = 0.2

# ---------------------------------------------------------------------------
# SPARK CONFIGURATION
# ---------------------------------------------------------------------------
SPARK_APP_NAME = "ScalableMarketBasketAnalysis"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = "8"

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
MIN_ITEMS_PER_TRANSACTION = 1
MAX_ITEMS_PER_TRANSACTION = 8


def ensure_directories():
    """Create the data/results/models directories if they do not exist."""
    for directory in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
