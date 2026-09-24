"""
performance_test.py
--------------------
Measures how the Spark FP-Growth pipeline scales as dataset size grows.

For each configured dataset size, this script:
    1. Generates (or reuses) a synthetic dataset of that size.
    2. Runs preprocessing (Spark) and times it.
    3. Runs FP-Growth (Spark) and times it.
    4. Records dataset size, transaction/product counts, timings,
       number of frequent itemsets, and number of association rules.

Results are written to results/performance_results.csv and a
matplotlib chart is saved to results/performance_analysis.png.

IMPORTANT: every number in the output CSV comes from an actual
measured run of the pipeline on this machine. Nothing is fabricated.
Because absolute timings depend heavily on the machine running this
script (CPU cores, RAM, disk speed), your numbers will differ from
anyone else's — that is expected and fine for a scalability study.

Usage:
    python src/performance_test.py
    python src/performance_test.py --sizes 10000 50000
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    DATA_DIR,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_SUPPORT,
    PERFORMANCE_PLOT_PATH,
    PERFORMANCE_RESULTS_PATH,
    PERFORMANCE_TEST_SIZES,
    ensure_directories,
)


def run_single_size_test(spark, size, min_support, min_confidence, keep_dataset=False):
    """
    Run the full generate -> preprocess -> FP-Growth pipeline for a
    single dataset size and return a dict of measured results.
    """
    from generate_dataset import generate_dataset, inject_minor_data_quality_issues
    from preprocess import preprocess_pipeline
    from fpgrowth_analysis import run_fpgrowth, extract_frequent_itemsets, extract_association_rules

    dataset_path = os.path.join(DATA_DIR, f"perf_test_{size}.csv")

    # --- Dataset generation (not timed as part of "processing time" since
    #     a real retailer would already have this data collected) ---
    if not os.path.exists(dataset_path):
        df = generate_dataset(size, seed=42)
        df = inject_minor_data_quality_issues(df, seed=42)
        df.to_csv(dataset_path, index=False)
    raw_row_count = sum(1 for _ in open(dataset_path)) - 1  # minus header

    # --- Preprocessing (Spark) ---
    preprocess_start = time.time()
    transactions_df, stats = preprocess_pipeline(spark, path=dataset_path, verbose=False)
    transactions_df.cache()
    num_transactions = transactions_df.count()  # forces evaluation
    preprocessing_time = time.time() - preprocess_start

    # --- FP-Growth (Spark MLlib) ---
    fpgrowth_start = time.time()
    model, fit_time = run_fpgrowth(transactions_df, min_support=min_support, min_confidence=min_confidence)
    itemsets_df = extract_frequent_itemsets(model, num_transactions)
    rules_df = extract_association_rules(model)
    fpgrowth_time = time.time() - fpgrowth_start

    total_time = preprocessing_time + fpgrowth_time

    result = {
        "dataset_size_transactions": size,
        "raw_row_count": raw_row_count,
        "num_transactions_after_cleaning": num_transactions,
        "num_unique_products": stats["num_unique_products"],
        "preprocessing_time_sec": round(preprocessing_time, 3),
        "fpgrowth_time_sec": round(fpgrowth_time, 3),
        "total_time_sec": round(total_time, 3),
        "num_frequent_itemsets": len(itemsets_df),
        "num_association_rules": len(rules_df),
        "throughput_txn_per_sec": round(num_transactions / total_time, 2) if total_time > 0 else None,
    }

    transactions_df.unpersist()

    if not keep_dataset:
        try:
            os.remove(dataset_path)
        except OSError:
            pass

    return result


def plot_results(results_df, output_path=PERFORMANCE_PLOT_PATH):
    """Generate a matplotlib figure showing scalability trends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    x = results_df["dataset_size_transactions"]

    axes[0, 0].plot(x, results_df["total_time_sec"], marker="o", color="#2563eb")
    axes[0, 0].set_title("Total Processing Time vs Dataset Size")
    axes[0, 0].set_xlabel("Number of Transactions")
    axes[0, 0].set_ylabel("Total Time (seconds)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, results_df["preprocessing_time_sec"], marker="o", label="Preprocessing", color="#16a34a")
    axes[0, 1].plot(x, results_df["fpgrowth_time_sec"], marker="o", label="FP-Growth", color="#dc2626")
    axes[0, 1].set_title("Preprocessing vs FP-Growth Time")
    axes[0, 1].set_xlabel("Number of Transactions")
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, results_df["throughput_txn_per_sec"], marker="o", color="#9333ea")
    axes[1, 0].set_title("Throughput vs Dataset Size")
    axes[1, 0].set_xlabel("Number of Transactions")
    axes[1, 0].set_ylabel("Transactions / Second")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, results_df["num_frequent_itemsets"], marker="o", label="Frequent Itemsets", color="#0891b2")
    axes[1, 1].plot(x, results_df["num_association_rules"], marker="o", label="Association Rules", color="#ea580c")
    axes[1, 1].set_title("Discovered Patterns vs Dataset Size")
    axes[1, 1].set_xlabel("Number of Transactions")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Scalable Market Basket Analysis — Performance & Scalability Analysis", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run scalability tests across multiple dataset sizes.")
    parser.add_argument("--sizes", type=int, nargs="+", default=PERFORMANCE_TEST_SIZES)
    parser.add_argument("--min_support", type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--keep_datasets", action="store_true", help="Keep generated test datasets on disk.")
    args = parser.parse_args()

    ensure_directories()

    from fpgrowth_analysis import get_spark_session

    spark = get_spark_session(app_name="PerformanceTest")

    results = []
    try:
        for size in args.sizes:
            print(f"\n=== Running performance test for {size:,} transactions ===")
            result = run_single_size_test(
                spark, size, args.min_support, args.min_confidence, keep_dataset=args.keep_datasets
            )
            for k, v in result.items():
                print(f"  {k}: {v}")
            results.append(result)
    finally:
        spark.stop()

    results_df = pd.DataFrame(results)
    results_df.to_csv(PERFORMANCE_RESULTS_PATH, index=False)
    print(f"\nSaved performance results to {PERFORMANCE_RESULTS_PATH}")

    plot_results(results_df)
    print(f"Saved performance chart to {PERFORMANCE_PLOT_PATH}")


if __name__ == "__main__":
    main()
