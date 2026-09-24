"""
preprocess.py
-------------
Spark-based data preprocessing pipeline for the Market Basket Analysis
project.

This module is intentionally implemented using Spark DataFrames (NOT
pandas) because the assignment requires the main data processing to be
distributed / scalable. Pandas is only used in generate_dataset.py for
creating the synthetic data and in business_insights.py for small
post-processing on already-aggregated (small) results.

Pipeline steps:
    1. Load the raw transaction CSV into a Spark DataFrame.
    2. Validate that required columns exist.
    3. Drop rows with missing / null product names.
    4. Drop duplicate (transaction_id, product_id) rows.
    5. Validate transaction IDs (must be non-null, non-empty).
    6. Validate product names (must be non-null, non-empty).
    7. Filter out invalid quantities (<= 0).
    8. Filter out invalid prices (<= 0).
    9. Group products by transaction_id into a list of items.
    10. Return a Spark DataFrame with columns: transaction_id, items
        (items is an array<string>) — the exact shape required by
        pyspark.ml.fpm.FPGrowth.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import RAW_DATASET_PATH


class DatasetError(Exception):
    """Raised when the input dataset is missing, empty, or malformed."""


REQUIRED_COLUMNS = [
    "transaction_id", "customer_id", "timestamp", "product_id",
    "product_name", "category", "quantity", "price",
]


def load_raw_dataframe(spark, path=RAW_DATASET_PATH):
    """
    Load the raw transaction CSV into a Spark DataFrame.

    Raises:
        DatasetError: if the file does not exist, is empty, or is
                      missing required columns.
    """
    if not os.path.exists(path):
        raise DatasetError(
            f"Dataset not found at '{path}'.\n"
            f"Run: python src/generate_dataset.py --num_transactions 10000"
        )

    if os.path.getsize(path) == 0:
        raise DatasetError(f"Dataset file '{path}' is empty.")

    df = spark.read.csv(path, header=True, inferSchema=True)

    if df.rdd.isEmpty():
        raise DatasetError(f"Dataset '{path}' contains a header but no data rows.")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise DatasetError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    return df


def clean_transactions(df):
    """
    Apply cleaning rules to the raw Spark DataFrame:
      - drop rows with null/empty transaction_id
      - drop rows with null/empty product_name
      - drop rows with invalid (<=0) quantity
      - drop rows with invalid (<=0) price
      - drop exact duplicate (transaction_id, product_id) pairs

    Returns a cleaned Spark DataFrame plus a small dict of counts
    describing how many rows were removed at each step (useful for
    logging / the pipeline report).
    """
    from pyspark.sql import functions as F

    stats = {"raw_row_count": df.count()}

    # 1. Validate transaction IDs
    df = df.filter(
        F.col("transaction_id").isNotNull() & (F.trim(F.col("transaction_id")) != "")
    )
    stats["after_transaction_id_validation"] = df.count()

    # 2. Validate product names (drop missing/blank product names)
    df = df.filter(
        F.col("product_name").isNotNull() & (F.trim(F.col("product_name")) != "")
    )
    stats["after_product_name_validation"] = df.count()

    # 3. Handle invalid quantities
    df = df.filter(F.col("quantity").isNotNull() & (F.col("quantity") > 0))
    stats["after_quantity_validation"] = df.count()

    # 4. Handle invalid prices
    df = df.filter(F.col("price").isNotNull() & (F.col("price") > 0))
    stats["after_price_validation"] = df.count()

    # 5. Remove duplicate (transaction_id, product_id) rows
    before_dedup = df.count()
    df = df.dropDuplicates(["transaction_id", "product_id"])
    stats["duplicates_removed"] = before_dedup - df.count()
    stats["clean_row_count"] = df.count()

    return df, stats


def aggregate_transactions(df):
    """
    Group the cleaned, row-per-product DataFrame into one row per
    transaction, with an `items` column containing the distinct list
    of product names purchased in that transaction.

    This is the exact shape pyspark.ml.fpm.FPGrowth expects:
        transaction_id | items
        T00000001      | [Bread, Butter, Milk]
    """
    from pyspark.sql import functions as F

    grouped = (
        df.groupBy("transaction_id")
        .agg(F.collect_set("product_name").alias("items"))
        .filter(F.size(F.col("items")) > 0)
    )
    return grouped


def preprocess_pipeline(spark, path=RAW_DATASET_PATH, verbose=True):
    """
    Run the full preprocessing pipeline end-to-end and return the
    transaction-level DataFrame (transaction_id, items) ready for
    FP-Growth, along with a stats dictionary.
    """
    raw_df = load_raw_dataframe(spark, path)
    clean_df, stats = clean_transactions(raw_df)

    if clean_df.rdd.isEmpty():
        raise DatasetError(
            "No valid rows remained after cleaning. Check the dataset for "
            "excessive missing values or invalid quantities/prices."
        )

    transactions_df = aggregate_transactions(clean_df)
    stats["num_transactions"] = transactions_df.count()
    stats["num_unique_products"] = (
        clean_df.select("product_name").distinct().count()
    )

    if verbose:
        print("Preprocessing summary:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

    return transactions_df, stats


if __name__ == "__main__":
    # Small standalone smoke test
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("PreprocessSmokeTest").master("local[*]").getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        txns_df, run_stats = preprocess_pipeline(spark)
        txns_df.show(5, truncate=80)
    except DatasetError as e:
        print(f"ERROR: {e}")
    finally:
        spark.stop()
