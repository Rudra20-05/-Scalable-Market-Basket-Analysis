"""
run_pipeline.py
-----------------
Master pipeline for the Scalable Market Basket Analysis project.

Runs, in order:
    1. Dataset generation (only if the dataset does not already exist)
    2. Spark initialization
    3. Data preprocessing (Spark)
    4. FP-Growth frequent pattern mining (Spark MLlib)
    5. Association rule extraction
    6. Business insight generation
    7. Model/algorithm evaluation summary (results/evaluation_summary.json)

All outputs are written to results/ so that the Streamlit dashboard can
simply read them without re-running the whole pipeline.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --num_transactions 50000 --min_support 0.02
    python src/run_pipeline.py --force_regenerate
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_NUM_TRANSACTIONS,
    EVALUATION_SUMMARY_PATH,
    PIPELINE_LOG_PATH,
    RAW_DATASET_PATH,
    ensure_directories,
)


def log(message, log_lines):
    print(message)
    log_lines.append(message)


def build_evaluation_summary(stats, fpgrowth_result, min_support, min_confidence, fit_time):
    """
    Build the evaluation summary for this frequent-pattern-mining task.

    Traditional classification metrics (accuracy, precision, recall) do
    NOT apply here because FP-Growth is an unsupervised pattern
    discovery algorithm — there is no ground-truth label to compare
    predictions against. Instead, the algorithm is evaluated using the
    metrics that are meaningful for association rule mining: how many
    patterns were found, how statistically significant they are
    (support/confidence/lift), and how long the mining took.
    """
    rules_df = fpgrowth_result["rules_df"]
    itemsets_df = fpgrowth_result["itemsets_df"]

    summary = {
        "algorithm": "Spark MLlib FP-Growth",
        "parameters": {
            "minSupport": min_support,
            "minConfidence": min_confidence,
        },
        "dataset": {
            "num_transactions": stats["num_transactions"],
            "num_unique_products": stats["num_unique_products"],
            "raw_row_count": stats["raw_row_count"],
            "clean_row_count": stats["clean_row_count"],
            "duplicates_removed": stats["duplicates_removed"],
        },
        "results": {
            "num_frequent_itemsets": int(len(itemsets_df)),
            "num_association_rules": int(len(rules_df)),
            "fpgrowth_fit_time_seconds": round(fit_time, 3),
        },
        "rule_quality": {
            "avg_support": round(float(rules_df["support"].mean()), 4) if not rules_df.empty else None,
            "avg_confidence": round(float(rules_df["confidence"].mean()), 4) if not rules_df.empty else None,
            "avg_lift": round(float(rules_df["lift"].mean()), 4) if not rules_df.empty else None,
            "max_lift": round(float(rules_df["lift"].max()), 4) if not rules_df.empty else None,
            "rules_with_lift_above_1": int((rules_df["lift"] > 1).sum()) if not rules_df.empty else 0,
        },
        "evaluation_notes": (
            "Accuracy, precision, and recall are not applicable to this task "
            "because FP-Growth is unsupervised frequent pattern mining with "
            "no ground-truth labels. Instead, rules are evaluated on "
            "statistical significance (support, confidence, lift) and the "
            "algorithm's runtime/scalability characteristics."
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the full Market Basket Analysis pipeline.")
    parser.add_argument("--num_transactions", type=int, default=DEFAULT_NUM_TRANSACTIONS)
    parser.add_argument("--dataset", type=str, default=RAW_DATASET_PATH)
    parser.add_argument("--min_support", type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--force_regenerate", action="store_true", help="Regenerate the dataset even if it already exists.")
    args = parser.parse_args()

    ensure_directories()
    log_lines = []
    pipeline_start = time.time()

    log("=" * 70, log_lines)
    log("SCALABLE MARKET BASKET ANALYSIS — PIPELINE RUN", log_lines)
    log("=" * 70, log_lines)

    # -----------------------------------------------------------------
    # Step 1: Dataset generation (only if needed)
    # -----------------------------------------------------------------
    if args.force_regenerate or not os.path.exists(args.dataset):
        log(f"\n[1/6] Generating dataset ({args.num_transactions:,} transactions) ...", log_lines)
        from generate_dataset import generate_dataset, inject_minor_data_quality_issues
        df = generate_dataset(args.num_transactions)
        df = inject_minor_data_quality_issues(df)
        os.makedirs(os.path.dirname(os.path.abspath(args.dataset)), exist_ok=True)
        df.to_csv(args.dataset, index=False)
        log(f"  Wrote {len(df):,} rows to {args.dataset}", log_lines)
    else:
        log(f"\n[1/6] Dataset already exists at {args.dataset} — skipping generation.", log_lines)
        log("  (use --force_regenerate to rebuild it)", log_lines)

    # -----------------------------------------------------------------
    # Step 2: Spark initialization
    # -----------------------------------------------------------------
    log("\n[2/6] Initializing Spark session (local[*]) ...", log_lines)
    from fpgrowth_analysis import get_spark_session, run_full_analysis
    from preprocess import preprocess_pipeline, DatasetError

    spark = get_spark_session()

    try:
        # -------------------------------------------------------------
        # Step 3: Preprocessing
        # -------------------------------------------------------------
        log("\n[3/6] Preprocessing data with Spark DataFrames ...", log_lines)
        try:
            transactions_df, stats = preprocess_pipeline(spark, path=args.dataset)
        except DatasetError as e:
            log(f"ERROR: {e}", log_lines)
            sys.exit(1)
        transactions_df.cache()

        # -------------------------------------------------------------
        # Step 4 + 5: FP-Growth + association rules
        # -------------------------------------------------------------
        log("\n[4/6] Running Spark MLlib FP-Growth ...", log_lines)
        result = run_full_analysis(
            spark, transactions_df, stats["num_transactions"],
            min_support=args.min_support, min_confidence=args.min_confidence,
        )

        if result["itemsets_df"].empty:
            log(
                "\nWARNING: No frequent itemsets were found with the current "
                "minSupport. Try lowering --min_support (e.g. 0.005).",
                log_lines,
            )

        # -------------------------------------------------------------
        # Step 6: Business insights
        # -------------------------------------------------------------
        log("\n[5/6] Generating business insights ...", log_lines)
        if not result["itemsets_df"].empty and not result["rules_df"].empty:
            from business_insights import generate_business_insights
            from config.config import BUSINESS_INSIGHTS_PATH
            insights_df = generate_business_insights(result["itemsets_df"], result["rules_df"])
            insights_df.to_csv(BUSINESS_INSIGHTS_PATH, index=False)
            log(f"  Generated {len(insights_df):,} insights -> {BUSINESS_INSIGHTS_PATH}", log_lines)
        else:
            log("  Skipped (no itemsets/rules to derive insights from).", log_lines)

        # -------------------------------------------------------------
        # Evaluation summary
        # -------------------------------------------------------------
        log("\n[6/6] Writing evaluation summary ...", log_lines)
        summary = build_evaluation_summary(
            stats, result, args.min_support, args.min_confidence, result["fit_time_seconds"]
        )
        with open(EVALUATION_SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"  Saved evaluation summary -> {EVALUATION_SUMMARY_PATH}", log_lines)

    finally:
        spark.stop()

    total_time = time.time() - pipeline_start
    log(f"\nPipeline complete in {total_time:.2f} seconds.", log_lines)
    log("Run the dashboard with: streamlit run dashboard/app.py", log_lines)

    with open(PIPELINE_LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
