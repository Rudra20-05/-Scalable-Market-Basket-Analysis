"""
generate_dataset.py
--------------------
Generates a realistic synthetic retail transaction dataset for the
Market Basket Analysis project.

Instead of assigning completely random, independent products to each
transaction, this generator uses a small set of "association seeds"
(defined in config/config.py) that bias certain products to co-occur
more often than chance (e.g. Bread -> Butter, Pasta -> Pasta Sauce).

FP-Growth is never given this seed information — it must rediscover
these relationships purely from the generated data. This keeps the
downstream frequent-pattern-mining results honest and meaningful
instead of trivially hard-coded.

Usage (PowerShell / CMD):
    python src/generate_dataset.py --num_transactions 10000
    python src/generate_dataset.py --num_transactions 500000 --output data/retail_transactions_500k.csv
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Allow running this script directly (python src/generate_dataset.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    ALL_PRODUCTS,
    ASSOCIATION_SEEDS,
    DEFAULT_NUM_TRANSACTIONS,
    MAX_ITEMS_PER_TRANSACTION,
    MIN_ITEMS_PER_TRANSACTION,
    PRODUCT_TO_CATEGORY,
    PRODUCT_TO_PRICE,
    RANDOM_SEED,
    RAW_DATASET_PATH,
    ensure_directories,
)


def build_companion_map():
    """
    Convert the flat ASSOCIATION_SEEDS list into a dict:
        trigger_product -> [(companion_product, probability), ...]
    """
    companion_map = {}
    for trigger, companion, probability in ASSOCIATION_SEEDS:
        companion_map.setdefault(trigger, []).append((companion, probability))
    return companion_map


def generate_basket(rng, companion_map):
    """
    Generate a single realistic shopping basket (a list of unique
    product names).

    Logic:
    1. Pick a random number of "base" products (uniformly at random).
    2. For every chosen product, check if it has known companion
       products (from ASSOCIATION_SEEDS). With the seeded probability,
       add the companion product to the basket too.
    3. Deduplicate and cap the basket size.
    """
    num_base_items = rng.randint(MIN_ITEMS_PER_TRANSACTION, MAX_ITEMS_PER_TRANSACTION - 2)
    basket = set(rng.sample(ALL_PRODUCTS, k=min(num_base_items, len(ALL_PRODUCTS))))

    # Add companion products based on seeded probabilities
    for product in list(basket):
        for companion, probability in companion_map.get(product, []):
            if rng.random() < probability:
                basket.add(companion)

    # Cap the basket size so it doesn't grow unbounded
    if len(basket) > MAX_ITEMS_PER_TRANSACTION:
        basket = set(rng.sample(sorted(basket), MAX_ITEMS_PER_TRANSACTION))

    return list(basket)


def generate_dataset(num_transactions, seed=RANDOM_SEED):
    """
    Generate `num_transactions` synthetic transactions and return them
    as a pandas DataFrame with one row per (transaction, product) pair
    — this "long format" mirrors what a real point-of-sale export
    typically looks like.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    companion_map = build_companion_map()

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_seconds = int((end_date - start_date).total_seconds())

    num_customers = max(200, num_transactions // 8)

    rows = []
    for txn_index in range(1, num_transactions + 1):
        transaction_id = f"T{txn_index:08d}"
        customer_id = f"C{rng.randint(1, num_customers):06d}"
        timestamp = start_date + timedelta(seconds=int(np_rng.integers(0, date_range_seconds)))

        basket = generate_basket(rng, companion_map)
        if not basket:
            # Extremely unlikely, but guarantee at least one item
            basket = [rng.choice(ALL_PRODUCTS)]

        for product_name in basket:
            quantity = rng.randint(1, 4)
            base_price = PRODUCT_TO_PRICE[product_name]
            # Small random price jitter to simulate real-world pricing variation
            price = round(base_price * rng.uniform(0.9, 1.1), 2)
            category = PRODUCT_TO_CATEGORY[product_name]
            product_id = f"P{ALL_PRODUCTS.index(product_name) + 1:03d}"

            rows.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "quantity": quantity,
                "price": price,
            })

        if txn_index % 50000 == 0:
            print(f"  ... generated {txn_index:,} / {num_transactions:,} transactions")

    df = pd.DataFrame(rows)
    return df


def inject_minor_data_quality_issues(df, seed=RANDOM_SEED):
    """
    Realistically, retail data is never perfectly clean. To make the
    preprocessing step (src/preprocess.py) meaningful, we intentionally
    inject a small number of data quality issues:
      - a handful of duplicate rows
      - a handful of missing product names
      - a handful of invalid (negative/zero) quantities and prices

    This affects at most ~0.5% of rows and is done AFTER the "clean"
    dataset is generated, so it never distorts the underlying
    association patterns used for FP-Growth evaluation.
    """
    rng = random.Random(seed + 1)
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    num_issues = max(1, int(n * 0.005))

    # Duplicate rows
    dup_indices = rng.sample(range(n), k=min(num_issues, n))
    duplicates = df.iloc[dup_indices]
    df = pd.concat([df, duplicates], ignore_index=True)

    # Missing product names
    n = len(df)
    missing_indices = rng.sample(range(n), k=min(num_issues, n))
    df.loc[missing_indices, "product_name"] = None

    # Invalid quantities (zero or negative)
    invalid_qty_indices = rng.sample(range(n), k=min(num_issues, n))
    df.loc[invalid_qty_indices, "quantity"] = rng.choice([0, -1, -3])

    # Invalid prices (negative)
    invalid_price_indices = rng.sample(range(n), k=min(num_issues, n))
    df.loc[invalid_price_indices, "price"] = -abs(df.loc[invalid_price_indices, "price"])

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic retail transaction dataset.")
    parser.add_argument(
        "--num_transactions", type=int, default=DEFAULT_NUM_TRANSACTIONS,
        help="Number of transactions (baskets) to generate.",
    )
    parser.add_argument(
        "--output", type=str, default=RAW_DATASET_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Skip injecting synthetic data-quality issues (produces a perfectly clean dataset).",
    )
    args = parser.parse_args()

    ensure_directories()

    print(f"Generating {args.num_transactions:,} transactions (seed={args.seed}) ...")
    df = generate_dataset(args.num_transactions, seed=args.seed)

    if not args.clean:
        df = inject_minor_data_quality_issues(df, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Done. Wrote {len(df):,} rows ({df['transaction_id'].nunique():,} unique transactions) to {args.output}")


if __name__ == "__main__":
    main()
