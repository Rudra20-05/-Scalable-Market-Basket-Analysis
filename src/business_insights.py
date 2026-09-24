"""
business_insights.py
---------------------
Turns the raw FP-Growth output (frequent itemsets + association rules)
into retailer-friendly business insights.

All insights are DERIVED from the actual results produced by
fpgrowth_analysis.py — nothing here is hard-coded. If the input files
don't exist yet, this script tells the user to run the pipeline first
instead of fabricating anything.

Usage (standalone):
    python src/business_insights.py
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    ASSOCIATION_RULES_PATH,
    BUSINESS_INSIGHTS_PATH,
    FREQUENT_ITEMSETS_PATH,
    PRODUCT_TO_CATEGORY,
    ensure_directories,
)


class InsightsError(Exception):
    """Raised when required upstream result files are missing."""


def _require_results():
    missing = [
        p for p in (FREQUENT_ITEMSETS_PATH, ASSOCIATION_RULES_PATH) if not os.path.exists(p)
    ]
    if missing:
        raise InsightsError(
            "Missing required result file(s): " + ", ".join(missing) + "\n"
            "Run: python src/run_pipeline.py"
        )


def _category_of(item_string):
    """
    Given a comma-separated item string like "Bread, Butter", return a
    comma-separated string of the categories those products belong to
    (deduplicated), using the PRODUCT_TO_CATEGORY lookup.
    """
    products = [p.strip() for p in item_string.split(",")]
    categories = sorted({PRODUCT_TO_CATEGORY.get(p, "Unknown") for p in products})
    return ", ".join(categories)


def top_frequent_combinations(itemsets_df, top_n=15):
    """Frequent itemsets with 2+ products, ranked by support."""
    multi = itemsets_df[itemsets_df["num_items"] >= 2].copy()
    return multi.sort_values("support", ascending=False).head(top_n)


def strong_associations(rules_df, top_n=15, min_lift=1.0):
    """Rules with lift > min_lift, ranked by lift (truly correlated, not coincidental)."""
    strong = rules_df[rules_df["lift"] > min_lift].copy()
    return strong.sort_values("lift", ascending=False).head(top_n)


def cross_selling_opportunities(rules_df, top_n=15, min_confidence=0.3, min_lift=1.2):
    """
    High-confidence, high-lift rules with a single-item antecedent are
    the most actionable for cross-selling ("if a customer buys X,
    recommend Y at checkout / nearby shelf placement").
    """
    single_antecedent = rules_df[~rules_df["antecedent"].str.contains(",")]
    candidates = single_antecedent[
        (single_antecedent["confidence"] >= min_confidence) &
        (single_antecedent["lift"] >= min_lift)
    ].copy()
    return candidates.sort_values(["lift", "confidence"], ascending=False).head(top_n)


def high_lift_pairs(rules_df, top_n=15):
    """Simple ranking of all rules purely by lift."""
    return rules_df.sort_values("lift", ascending=False).head(top_n)


def category_level_relationships(rules_df, top_n=15):
    """
    Aggregate association rules up to the category level to reveal
    higher-level merchandising insights (e.g. Dairy <-> Grocery).
    """
    df = rules_df.copy()
    df["antecedent_category"] = df["antecedent"].apply(_category_of)
    df["consequent_category"] = df["consequent"].apply(_category_of)

    # Drop rules where antecedent and consequent map to the same category
    df = df[df["antecedent_category"] != df["consequent_category"]]

    grouped = (
        df.groupby(["antecedent_category", "consequent_category"])
        .agg(avg_confidence=("confidence", "mean"), avg_lift=("lift", "mean"), num_rules=("lift", "count"))
        .reset_index()
        .sort_values("avg_lift", ascending=False)
    )
    return grouped.head(top_n)


def generate_business_insights(itemsets_df, rules_df):
    """
    Build a single tidy DataFrame combining several insight categories,
    tagged by an `insight_type` column, suitable for a CSV export and
    for the dashboard to filter on.
    """
    sections = []

    combos = top_frequent_combinations(itemsets_df)
    for _, row in combos.iterrows():
        sections.append({
            "insight_type": "Frequent Combination",
            "detail": row["items"],
            "metric": "support",
            "value": round(row["support"], 4),
        })

    strong = strong_associations(rules_df)
    for _, row in strong.iterrows():
        sections.append({
            "insight_type": "Strong Association",
            "detail": f"{row['antecedent']} -> {row['consequent']}",
            "metric": "lift",
            "value": round(row["lift"], 4),
        })

    cross_sell = cross_selling_opportunities(rules_df)
    for _, row in cross_sell.iterrows():
        sections.append({
            "insight_type": "Cross-Selling Opportunity",
            "detail": f"{row['antecedent']} -> {row['consequent']}",
            "metric": "confidence",
            "value": round(row["confidence"], 4),
        })

    high_lift = high_lift_pairs(rules_df)
    for _, row in high_lift.iterrows():
        sections.append({
            "insight_type": "High Lift Pair",
            "detail": f"{row['antecedent']} -> {row['consequent']}",
            "metric": "lift",
            "value": round(row["lift"], 4),
        })

    category_rel = category_level_relationships(rules_df)
    for _, row in category_rel.iterrows():
        sections.append({
            "insight_type": "Category-Level Relationship",
            "detail": f"{row['antecedent_category']} -> {row['consequent_category']}",
            "metric": "avg_lift",
            "value": round(row["avg_lift"], 4),
        })

    return pd.DataFrame(sections)


def main():
    ensure_directories()
    _require_results()

    itemsets_df = pd.read_csv(FREQUENT_ITEMSETS_PATH)
    rules_df = pd.read_csv(ASSOCIATION_RULES_PATH)

    if itemsets_df.empty or rules_df.empty:
        raise InsightsError(
            "Frequent itemsets or association rules are empty. Try lowering "
            "minSupport / minConfidence and re-run src/run_pipeline.py."
        )

    insights_df = generate_business_insights(itemsets_df, rules_df)
    insights_df.to_csv(BUSINESS_INSIGHTS_PATH, index=False)

    print(f"Generated {len(insights_df):,} business insights.")
    print(f"Saved to {BUSINESS_INSIGHTS_PATH}")
    print("\nInsight type breakdown:")
    print(insights_df["insight_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
