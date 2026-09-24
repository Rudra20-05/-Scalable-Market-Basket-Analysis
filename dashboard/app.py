"""
dashboard/app.py
-----------------
Streamlit dashboard for the Scalable Market Basket Analysis project.

IMPORTANT: This dashboard only READS already-generated result files
from results/. It does not re-run Spark/FP-Growth on every interaction
— run `python src/run_pipeline.py` first to generate the data this
dashboard displays.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    ASSOCIATION_RULES_PATH,
    BUSINESS_INSIGHTS_PATH,
    EVALUATION_SUMMARY_PATH,
    FREQUENT_ITEMSETS_PATH,
    PERFORMANCE_RESULTS_PATH,
    PRODUCT_TO_CATEGORY,
    RAW_DATASET_PATH,
)

st.set_page_config(
    page_title="Scalable Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
)

RUN_PIPELINE_MSG = (
    "Results not found. Run `python src/run_pipeline.py` from the project "
    "root to generate them, then refresh this page."
)


# ---------------------------------------------------------------------------
# Data loading (cached so the dashboard stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv_if_exists(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


itemsets_df = load_csv_if_exists(FREQUENT_ITEMSETS_PATH)
rules_df = load_csv_if_exists(ASSOCIATION_RULES_PATH)
insights_df = load_csv_if_exists(BUSINESS_INSIGHTS_PATH)
performance_df = load_csv_if_exists(PERFORMANCE_RESULTS_PATH)
evaluation_summary = load_json_if_exists(EVALUATION_SUMMARY_PATH)
raw_dataset_preview = load_csv_if_exists(RAW_DATASET_PATH)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 Scalable Market Basket Analysis Dashboard")
st.caption(
    "Frequent pattern mining on retail transactions using **Apache Spark MLlib FP-Growth**"
)

if itemsets_df is None or rules_df is None:
    st.error(RUN_PIPELINE_MSG)
    st.stop()


# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
st.header("📊 Overview")

total_transactions = None
total_products = None
if raw_dataset_preview is not None:
    total_transactions = raw_dataset_preview["transaction_id"].nunique()
    total_products = raw_dataset_preview["product_name"].nunique()
elif evaluation_summary is not None:
    total_transactions = evaluation_summary["dataset"]["num_transactions"]
    total_products = evaluation_summary["dataset"]["num_unique_products"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_transactions:,}" if total_transactions else "N/A")
col2.metric("Unique Products", f"{total_products:,}" if total_products else "N/A")
col3.metric("Frequent Itemsets", f"{len(itemsets_df):,}")
col4.metric("Association Rules", f"{len(rules_df):,}")

if evaluation_summary is not None:
    with st.expander("Algorithm parameters used for this run"):
        st.json(evaluation_summary["parameters"])

st.divider()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Rule Filters")
min_support_filter = st.sidebar.slider(
    "Minimum Support", 0.0, float(rules_df["support"].max()), 0.0, step=0.001, format="%.3f"
)
min_confidence_filter = st.sidebar.slider(
    "Minimum Confidence", 0.0, 1.0, 0.0, step=0.01
)
min_lift_filter = st.sidebar.slider(
    "Minimum Lift", 0.0, float(max(rules_df["lift"].max(), 1.0)), 0.0, step=0.1
)

filtered_rules_df = rules_df[
    (rules_df["support"] >= min_support_filter) &
    (rules_df["confidence"] >= min_confidence_filter) &
    (rules_df["lift"] >= min_lift_filter)
].copy()

st.sidebar.caption(f"Showing {len(filtered_rules_df):,} of {len(rules_df):,} rules")


# ---------------------------------------------------------------------------
# Frequent Itemsets
# ---------------------------------------------------------------------------
st.header("🧺 Frequent Itemsets")
sort_col = st.selectbox("Sort itemsets by", ["support", "freq", "num_items"], index=0)
st.dataframe(
    itemsets_df.sort_values(sort_col, ascending=False),
    use_container_width=True,
    height=320,
)

st.divider()


# ---------------------------------------------------------------------------
# Association Rules
# ---------------------------------------------------------------------------
st.header("🔗 Association Rules")
st.dataframe(
    filtered_rules_df.sort_values("lift", ascending=False),
    use_container_width=True,
    height=320,
)

st.divider()


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.header("📈 Visualizations")

viz_tabs = st.tabs([
    "Top Products", "Top Itemsets", "Support vs Confidence",
    "Confidence vs Lift", "Association Network", "Category Relationships",
    "Cross-Selling",
])

# --- Top Frequent Products ---
with viz_tabs[0]:
    single_items = itemsets_df[itemsets_df["num_items"] == 1].copy()
    single_items = single_items.sort_values("support", ascending=False).head(15)
    fig = px.bar(
        single_items, x="items", y="support",
        title="Top 15 Most Frequently Purchased Individual Products",
        labels={"items": "Product", "support": "Support"},
        color="support", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Top Frequent Itemsets ---
with viz_tabs[1]:
    multi_items = itemsets_df[itemsets_df["num_items"] >= 2].copy()
    multi_items = multi_items.sort_values("support", ascending=False).head(15)
    if multi_items.empty:
        st.info("No multi-item itemsets found at the current minSupport threshold.")
    else:
        fig = px.bar(
            multi_items, x="support", y="items", orientation="h",
            title="Top 15 Frequent Product Combinations",
            labels={"items": "Product Combination", "support": "Support"},
            color="support", color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# --- Support vs Confidence scatter ---
with viz_tabs[2]:
    if filtered_rules_df.empty:
        st.info("No rules match the current filters.")
    else:
        fig = px.scatter(
            filtered_rules_df, x="support", y="confidence", size="lift", color="lift",
            hover_data=["antecedent", "consequent"],
            title="Support vs Confidence (bubble size/color = lift)",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Confidence vs Lift scatter ---
with viz_tabs[3]:
    if filtered_rules_df.empty:
        st.info("No rules match the current filters.")
    else:
        fig = px.scatter(
            filtered_rules_df, x="confidence", y="lift", size="support", color="support",
            hover_data=["antecedent", "consequent"],
            title="Confidence vs Lift (bubble size/color = support)",
            color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Association Network ---
with viz_tabs[4]:
    st.caption("Nodes = products, edges = association rules (edge width = lift)")
    top_n_network = st.slider("Number of top rules (by lift) to visualize", 5, 50, 20, key="network_slider")

    try:
        import networkx as nx

        network_rules = rules_df.sort_values("lift", ascending=False).head(top_n_network)
        # Only visualize single-item -> single-item rules for a readable graph
        network_rules = network_rules[
            (~network_rules["antecedent"].str.contains(",")) &
            (~network_rules["consequent"].str.contains(","))
        ]

        if network_rules.empty:
            st.info("Not enough simple (single-item) rules to build a network graph.")
        else:
            G = nx.DiGraph()
            for _, row in network_rules.iterrows():
                G.add_edge(row["antecedent"], row["consequent"], weight=row["lift"])

            pos = nx.spring_layout(G, seed=42, k=0.8)

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, line=dict(width=1.5, color="#94a3b8"),
                hoverinfo="none", mode="lines",
            )

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()),
                textposition="top center",
                marker=dict(size=22, color="#2563eb", line=dict(width=2, color="white")),
                hoverinfo="text",
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Product Association Network (Top Rules by Lift)",
                showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning(
            "NetworkX is not installed, so the network graph is unavailable. "
            "Install it with `pip install networkx` to enable this view."
        )

# --- Category-level relationships ---
with viz_tabs[5]:
    rules_with_cat = rules_df.copy()

    def _category_str(item_string):
        products = [p.strip() for p in str(item_string).split(",")]
        cats = sorted({PRODUCT_TO_CATEGORY.get(p, "Unknown") for p in products})
        return ", ".join(cats)

    rules_with_cat["antecedent_category"] = rules_with_cat["antecedent"].apply(_category_str)
    rules_with_cat["consequent_category"] = rules_with_cat["consequent"].apply(_category_str)
    rules_with_cat = rules_with_cat[
        rules_with_cat["antecedent_category"] != rules_with_cat["consequent_category"]
    ]

    if rules_with_cat.empty:
        st.info("No cross-category rules found at the current thresholds.")
    else:
        cat_summary = (
            rules_with_cat.groupby(["antecedent_category", "consequent_category"])
            .agg(avg_lift=("lift", "mean"), num_rules=("lift", "count"))
            .reset_index()
        )
        fig = px.density_heatmap(
            cat_summary, x="antecedent_category", y="consequent_category", z="avg_lift",
            title="Average Lift Between Product Categories",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Cross-selling opportunities ---
with viz_tabs[6]:
    single_antecedent_rules = rules_df[~rules_df["antecedent"].str.contains(",")].copy()
    top_cross_sell = single_antecedent_rules.sort_values(
        ["lift", "confidence"], ascending=False
    ).head(15)
    if top_cross_sell.empty:
        st.info("No cross-selling candidates found at the current thresholds.")
    else:
        top_cross_sell["rule"] = top_cross_sell["antecedent"] + " → " + top_cross_sell["consequent"]
        fig = px.bar(
            top_cross_sell, x="lift", y="rule", orientation="h", color="confidence",
            title="Top 15 Cross-Selling Opportunities (by lift)",
            color_continuous_scale="Teal",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Product search
# ---------------------------------------------------------------------------
st.header("🔍 Product Search")
st.caption("Select a product to see what it's associated with, and what to recommend alongside it.")

all_single_products = sorted(itemsets_df[itemsets_df["num_items"] == 1]["items"].unique())

if not all_single_products:
    st.info("No single-item frequent itemsets available to search.")
else:
    selected_product = st.selectbox("Choose a product", all_single_products)

    matching_rules = rules_df[
        rules_df["antecedent"].apply(
            lambda a: selected_product in [p.strip() for p in a.split(",")]
        )
    ].sort_values(["lift", "confidence"], ascending=False)

    if matching_rules.empty:
        st.info(f"No association rules found where '{selected_product}' is in the antecedent at the current global thresholds.")
    else:
        st.write(f"**Recommended pairings for customers buying `{selected_product}`:**")
        for _, row in matching_rules.head(10).iterrows():
            st.markdown(
                f"- **{row['antecedent']} → {row['consequent']}**  "
                f"(support={row['support']:.3f}, confidence={row['confidence']:.3f}, lift={row['lift']:.3f})"
            )

st.divider()


# ---------------------------------------------------------------------------
# Business Insights
# ---------------------------------------------------------------------------
st.header("💡 Business Insights")
if insights_df is None:
    st.info("Business insights not found. Run `python src/business_insights.py` to generate them.")
else:
    insight_type_filter = st.multiselect(
        "Filter by insight type",
        options=sorted(insights_df["insight_type"].unique()),
        default=sorted(insights_df["insight_type"].unique()),
    )
    st.dataframe(
        insights_df[insights_df["insight_type"].isin(insight_type_filter)],
        use_container_width=True,
        height=350,
    )

st.divider()


# ---------------------------------------------------------------------------
# Performance / Scalability
# ---------------------------------------------------------------------------
st.header("⚡ Performance & Scalability")
if performance_df is None:
    st.info("Performance results not found. Run `python src/performance_test.py` to generate them.")
else:
    st.dataframe(performance_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            performance_df, x="dataset_size_transactions", y="total_time_sec",
            markers=True, title="Total Processing Time vs Dataset Size",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(
            performance_df, x="dataset_size_transactions", y="throughput_txn_per_sec",
            markers=True, title="Throughput (transactions/sec) vs Dataset Size",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

if evaluation_summary is not None:
    with st.expander("📋 Full Evaluation Summary (JSON)"):
        st.json(evaluation_summary)

st.caption(
    "Built with Apache Spark MLlib FP-Growth • Streamlit • Plotly — "
    "Scalable Market Basket Analysis mini-project"
)
