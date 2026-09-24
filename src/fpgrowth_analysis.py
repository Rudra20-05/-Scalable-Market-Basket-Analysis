"""
fpgrowth_analysis.py
---------------------
Runs Spark MLlib's FP-Growth algorithm (pyspark.ml.fpm.FPGrowth) on the
preprocessed transaction data to discover frequent itemsets and
association rules.

This is the core "scalable frequent pattern mining" component of the
project. It genuinely uses Spark's distributed FP-Growth implementation
— no shortcuts, no re-implementation in pure Python/Pandas.

Usage (standalone):
    python src/fpgrowth_analysis.py --min_support 0.01 --min_confidence 0.2
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    ASSOCIATION_RULES_PATH,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_SUPPORT,
    FREQUENT_ITEMSETS_PATH,
    RAW_DATASET_PATH,
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
    ensure_directories,
)


class FPGrowthConfigError(Exception):
    """Raised when minSupport / minConfidence parameters are invalid."""


def get_spark_session(app_name=SPARK_APP_NAME):
    """
    Create (or fetch) a local Spark session configured for this project.

    Uses local[*] so Spark utilizes all available CPU cores on the
    machine — this is what makes the pipeline "distributed processing"
    even when run on a single laptop, and it will transparently scale
    to a real cluster by changing SPARK_MASTER in config/config.py.
    """
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName(app_name)
            .master(SPARK_MASTER)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Failed to initialize Spark session. Make sure Java is "
            "installed and JAVA_HOME is set correctly. See setup_windows.md. "
            f"Original error: {exc}"
        )


def validate_fpgrowth_params(min_support, min_confidence):
    if not (0 < min_support <= 1):
        raise FPGrowthConfigError(f"minSupport must be in (0, 1]. Got {min_support}.")
    if not (0 <= min_confidence <= 1):
        raise FPGrowthConfigError(f"minConfidence must be in [0, 1]. Got {min_confidence}.")


def run_fpgrowth(transactions_df, min_support=DEFAULT_MIN_SUPPORT,
                  min_confidence=DEFAULT_MIN_CONFIDENCE):
    """
    Fit Spark MLlib's FPGrowth model on the transactions DataFrame.

    Args:
        transactions_df: Spark DataFrame with columns (transaction_id, items)
        min_support: minimum support threshold for an itemset to be frequent
        min_confidence: minimum confidence threshold for an association rule

    Returns:
        model: the fitted pyspark.ml.fpm.FPGrowthModel
        fit_time_seconds: wall-clock time taken to fit the model
    """
    from pyspark.ml.fpm import FPGrowth

    validate_fpgrowth_params(min_support, min_confidence)

    fp_growth = FPGrowth(
        itemsCol="items",
        minSupport=min_support,
        minConfidence=min_confidence,
    )

    start = time.time()
    model = fp_growth.fit(transactions_df)
    fit_time_seconds = time.time() - start

    return model, fit_time_seconds


def extract_frequent_itemsets(model, num_transactions):
    """
    Extract the frequent itemsets from a fitted FPGrowthModel as a
    clean pandas DataFrame with columns: items, freq, support.
    """
    itemsets_df = model.freqItemsets.toPandas()

    if itemsets_df.empty:
        return itemsets_df

    # items column comes back as a Python list already (Spark array<string>)
    itemsets_df["items"] = itemsets_df["items"].apply(lambda x: ", ".join(sorted(x)))
    itemsets_df["num_items"] = itemsets_df["items"].apply(lambda x: len(x.split(", ")))
    itemsets_df["support"] = itemsets_df["freq"] / num_transactions

    itemsets_df = itemsets_df.sort_values("support", ascending=False).reset_index(drop=True)
    return itemsets_df[["items", "num_items", "freq", "support"]]


def extract_association_rules(model):
    """
    Extract association rules from a fitted FPGrowthModel as a clean
    pandas DataFrame with columns:
        antecedent, consequent, confidence, lift, support
    """
    rules_df = model.associationRules.toPandas()

    if rules_df.empty:
        return rules_df

    rules_df["antecedent"] = rules_df["antecedent"].apply(lambda x: ", ".join(sorted(x)))
    rules_df["consequent"] = rules_df["consequent"].apply(lambda x: ", ".join(sorted(x)))

    rules_df = rules_df.rename(columns={"confidence": "confidence", "lift": "lift"})
    rules_df = rules_df[["antecedent", "consequent", "confidence", "lift", "support"]]
    rules_df = rules_df.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

    return rules_df


def run_full_analysis(spark, transactions_df, num_transactions,
                       min_support=DEFAULT_MIN_SUPPORT,
                       min_confidence=DEFAULT_MIN_CONFIDENCE,
                       save=True, verbose=True):
    """
    Convenience wrapper: fits FP-Growth, extracts itemsets + rules,
    optionally saves them to results/, and returns everything the
    caller (pipeline / dashboard / performance test) might need.
    """
    ensure_directories()

    model, fit_time_seconds = run_fpgrowth(
        transactions_df, min_support=min_support, min_confidence=min_confidence
    )

    itemsets_df = extract_frequent_itemsets(model, num_transactions)
    rules_df = extract_association_rules(model)

    if verbose:
        print(f"FP-Growth fit time: {fit_time_seconds:.2f}s")
        print(f"Frequent itemsets found: {len(itemsets_df):,}")
        print(f"Association rules found: {len(rules_df):,}")

    if save:
        itemsets_df.to_csv(FREQUENT_ITEMSETS_PATH, index=False)
        rules_df.to_csv(ASSOCIATION_RULES_PATH, index=False)
        if verbose:
            print(f"Saved frequent itemsets to {FREQUENT_ITEMSETS_PATH}")
            print(f"Saved association rules to {ASSOCIATION_RULES_PATH}")

    return {
        "model": model,
        "itemsets_df": itemsets_df,
        "rules_df": rules_df,
        "fit_time_seconds": fit_time_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Spark MLlib FP-Growth on the retail transaction dataset.")
    parser.add_argument("--dataset", type=str, default=RAW_DATASET_PATH)
    parser.add_argument("--min_support", type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    args = parser.parse_args()

    from preprocess import preprocess_pipeline

    spark = get_spark_session()
    try:
        transactions_df, stats = preprocess_pipeline(spark, path=args.dataset)
        transactions_df.cache()
        num_transactions = stats["num_transactions"]

        result = run_full_analysis(
            spark, transactions_df, num_transactions,
            min_support=args.min_support, min_confidence=args.min_confidence,
        )

        print("\nTop 10 association rules by lift:")
        print(result["rules_df"].head(10).to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
