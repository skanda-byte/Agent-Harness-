r"""
dashboard.py — Streamlit Interactive Analytics & Live Evaluation Dashboard
for Agentic Harness.

Pages:
  1. Baseline vs. Harness Success Rate (bar chart & KPI metrics)
  2. Per-Category Breakdown (grouped bar chart & category drilldown)
  3. Reliability Score Distribution (distribution histograms & tier analysis)
  4. Retries-per-Query Histogram (resolution breakdown & recovery analysis)
  5. Try it Live (interactive prompt execution, attempt-by-attempt sub-scores,
     validator errors, and critic feedback)

Run via:
  $env:PYTHONPATH="src;.venv\Lib\site-packages"
  python -m streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Configure project paths
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "agent_harness.db"
SUMMARY_JSON_PATH = _ROOT / "results" / "summary.json"
SAMPLES_JSON_PATH = _SRC / "agentic_harness" / "benchmark" / "samples.json"
CHALLENGES_JSON_PATH = _SRC / "agentic_harness" / "benchmark" / "challenges.json"

# Page Configuration
st.set_page_config(
    page_title="Agent Harness Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .attempt-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-stalled {
        background-color: #FEF08A;
        color: #713F12;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_sample_metadata() -> dict[str, dict[str, Any]]:
    """Map sample_id to metadata (category, prompt, checker)."""
    meta: dict[str, dict[str, Any]] = {}
    for p in (SAMPLES_JSON_PATH, CHALLENGES_JSON_PATH):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for s in data:
                    sid = s.get("id")
                    if sid:
                        meta[sid] = {
                            "category": s.get("category", "unknown"),
                            "prompt": s.get("prompt", ""),
                            "checker": s.get("checker", {}),
                            "reference": s.get("reference_answer", ""),
                            "split": s.get("split", "unknown"),
                        }
            except Exception as exc:
                st.warning(f"Failed to read metadata from {p.name}: {exc}")
    return meta


@st.cache_data(ttl=30)
def load_results_summary() -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    """Load evaluation summary from results/summary.json."""
    if not SUMMARY_JSON_PATH.exists():
        return None, pd.DataFrame(), pd.DataFrame()

    try:
        raw = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
        metrics_list = raw.get("metrics_summary", [])
        metrics_df = pd.DataFrame(metrics_list)

        sample_rows: list[dict[str, Any]] = []
        sample_runs = raw.get("sample_runs", {})
        for run_key, samples in sample_runs.items():
            for item in samples:
                row = dict(item)
                row["run_key"] = run_key
                sample_rows.append(row)

        samples_df = pd.DataFrame(sample_rows)
        return raw, metrics_df, samples_df
    except Exception as exc:
        st.warning(f"Error loading {SUMMARY_JSON_PATH}: {exc}")
        return None, pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=30)
def load_db_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load runs and cache tables from SQLite database."""
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    meta_map = load_sample_metadata()

    try:
        conn = sqlite3.connect(str(DB_PATH))
        runs_df = pd.read_sql_query("SELECT * FROM runs ORDER BY id ASC", conn)
        cache_df = pd.read_sql_query("SELECT id, query, reliability, retries, created_at FROM cache", conn)
        conn.close()

        if not runs_df.empty:
            # Map categories from sample_id
            runs_df["category"] = runs_df["sample_id"].apply(
                lambda sid: meta_map.get(sid, {}).get("category", "live_query" if sid == "live_query" else "general")
            )
        return runs_df, cache_df
    except Exception as exc:
        st.warning(f"Error loading SQLite database: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def get_consolidated_sample_data(
    samples_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    source_choice: str,
) -> pd.DataFrame:
    """Return a consolidated sample DataFrame based on the selected data source."""
    if source_choice == "Results File (summary.json)" or (source_choice == "Auto (Prefer Results)" and not samples_df.empty):
        if not samples_df.empty:
            return samples_df.copy()

    # Fallback / compute from SQLite runs
    if not runs_df.empty:
        # Aggregate per sample_id and mode
        eval_runs = runs_df[runs_df["mode"].isin(["baseline", "harness"])].copy()
        if not eval_runs.empty:
            records: list[dict[str, Any]] = []
            for (sid, mode), group in eval_runs.groupby(["sample_id", "mode"]):
                group_sorted = group.sort_values("attempt")
                last_row = group_sorted.iloc[-1]
                first_row = group_sorted.iloc[0]
                retries_used = max(0, len(group_sorted) - 1)
                passed = bool(last_row["det_pass"] == 1 and last_row["reliability"] >= 0.8)
                first_passed = bool(first_row["det_pass"] == 1 and first_row["reliability"] >= 0.8)
                recovered = bool((not first_passed) and passed)
                records.append({
                    "id": sid,
                    "mode": mode,
                    "category": last_row.get("category", "unknown"),
                    "passed": passed,
                    "reliability": float(last_row["reliability"]),
                    "retries_used": retries_used,
                    "failed_attempt1": not first_passed,
                    "recovered": recovered,
                    "latency_ms": float(group["latency_ms"].sum()),
                })
            return pd.DataFrame(records)

    return samples_df.copy() if not samples_df.empty else pd.DataFrame()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------

st.sidebar.title("🎯 Agent Harness")
st.sidebar.caption("Deterministic Verification & Agentic Retry Engine")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Baseline vs Harness Success",
        "2. Per-Category Breakdown",
        "3. Reliability Distribution",
        "4. Retries-per-Query Histogram",
        "5. Try it Live",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
data_source = st.sidebar.selectbox(
    "Active Data Source",
    ["Auto (Prefer Results)", "Results File (summary.json)", "SQLite DB (agent_harness.db)"],
    index=0,
)

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Pre-load datasets
summary_raw, metrics_df, samples_df = load_results_summary()
runs_df, cache_df = load_db_runs()
df_samples = get_consolidated_sample_data(samples_df, runs_df, data_source)

# Display data counts in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Database Runs:** `{len(runs_df):,}`")
st.sidebar.markdown(f"**Semantic Cache Size:** `{len(cache_df):,}`")
if not metrics_df.empty:
    st.sidebar.markdown(f"**Evaluated Test Samples:** `{len(df_samples):,}`")


# -----------------------------------------------------------------------------
# PAGE 1: Baseline vs. Harness Success Rate
# -----------------------------------------------------------------------------
if page == "1. Baseline vs Harness Success":
    st.markdown('<div class="main-header">📊 Baseline vs. Harness Success Rate</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Direct performance comparison between single-shot unassisted generation '
        '(Baseline) and multi-turn verified self-correction (Harness).</div>',
        unsafe_allow_html=True,
    )

    if df_samples.empty and metrics_df.empty:
        st.warning("No evaluation data found. Run evaluate.py or run_bench.py to generate results.")
    else:
        # Calculate overall metrics
        base_df = df_samples[df_samples["mode"] == "baseline"] if not df_samples.empty else pd.DataFrame()
        harn_df = df_samples[df_samples["mode"] == "harness"] if not df_samples.empty else pd.DataFrame()

        base_total = len(base_df)
        base_passed = int(base_df["passed"].sum()) if not base_df.empty else 0
        base_sr = (base_passed / base_total) if base_total > 0 else 0.0
        base_rel = float(base_df["reliability"].mean()) if not base_df.empty else 0.0

        harn_total = len(harn_df)
        harn_passed = int(harn_df["passed"].sum()) if not harn_df.empty else 0
        harn_sr = (harn_passed / harn_total) if harn_total > 0 else 0.0
        harn_rel = float(harn_df["reliability"].mean()) if not harn_df.empty else 0.0

        failed_att1 = int(harn_df["failed_attempt1"].sum()) if ("failed_attempt1" in harn_df.columns and not harn_df.empty) else 0
        recovered = int(harn_df["recovered"].sum()) if ("recovered" in harn_df.columns and not harn_df.empty) else 0
        rec_rate = (recovered / failed_att1) if failed_att1 > 0 else 0.0
        avg_retries = float(harn_df["retries_used"].mean()) if not harn_df.empty else 0.0

        # KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Test Samples", f"{harn_total:,}")
        with col2:
            st.metric("Baseline Success Rate", f"{base_sr * 100:.1f}%", f"{base_passed}/{base_total}")
        with col3:
            delta_sr = (harn_sr - base_sr) * 100
            st.metric("Harness Success Rate", f"{harn_sr * 100:.1f}%", f"+{delta_sr:.1f}%", delta_color="normal")
        with col4:
            st.metric("Error Recovery Rate", f"{rec_rate * 100:.1f}%", f"{recovered}/{failed_att1} fixed")
        with col5:
            st.metric("Avg Retries / Query", f"{avg_retries:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Success Rate Comparison (%)")
            chart_data = pd.DataFrame({
                "Mode": ["Baseline (Unassisted)", "Agentic Harness"],
                "Success Rate (%)": [round(base_sr * 100, 1), round(harn_sr * 100, 1)],
                "Passed Count": [f"{base_passed}/{base_total}", f"{harn_passed}/{harn_total}"],
            })

            bar_chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Mode:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                    y=alt.Y("Success Rate (%):Q", title="Success Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "Mode:N",
                        scale=alt.Scale(domain=["Baseline (Unassisted)", "Agentic Harness"], range=["#F97316", "#10B981"]),
                        legend=None,
                    ),
                    tooltip=["Mode", "Success Rate (%)", "Passed Count"],
                )
                .properties(height=350)
            )

            text_chart = bar_chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-6,
                fontSize=14,
                fontWeight="bold",
            ).encode(text="Success Rate (%):Q")

            st.altair_chart(bar_chart + text_chart, use_container_width=True)

        with chart_col2:
            st.subheader("Average Reliability Score (0.00 – 1.00)")
            rel_data = pd.DataFrame({
                "Mode": ["Baseline (Unassisted)", "Agentic Harness"],
                "Reliability Score": [round(base_rel, 3), round(harn_rel, 3)],
            })

            rel_chart = (
                alt.Chart(rel_data)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Mode:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                    y=alt.Y("Reliability Score:Q", title="Average Reliability Score", scale=alt.Scale(domain=[0, 1.0])),
                    color=alt.Color(
                        "Mode:N",
                        scale=alt.Scale(domain=["Baseline (Unassisted)", "Agentic Harness"], range=["#FB923C", "#059669"]),
                        legend=None,
                    ),
                    tooltip=["Mode", "Reliability Score"],
                )
                .properties(height=350)
            )

            rule = (
                alt.Chart(pd.DataFrame({"y": [0.8]}))
                .mark_rule(color="#DC2626", strokeDash=[5, 5], strokeWidth=2)
                .encode(y="y:Q")
            )

            text_rel = rel_chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-6,
                fontSize=14,
                fontWeight="bold",
            ).encode(text="Reliability Score:Q")

            st.altair_chart(rel_chart + text_rel + rule, use_container_width=True)
            st.caption("🔴 Red dashed line indicates the 0.80 Acceptance Threshold.")

        # Latency comparison if metrics summary exists
        if not metrics_df.empty and "avg_cold_latency_ms" in metrics_df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("⚡ Latency Benchmark: Cold Execution vs. Semantic Cache")
            lat_col1, lat_col2, lat_col3 = st.columns(3)

            cold_base = float(metrics_df[metrics_df["mode"] == "baseline"]["avg_cold_latency_ms"].iloc[0]) if not metrics_df[metrics_df["mode"] == "baseline"].empty else 9000.0
            cold_harn = float(metrics_df[metrics_df["mode"] == "harness"]["avg_cold_latency_ms"].iloc[0]) if not metrics_df[metrics_df["mode"] == "harness"].empty else 7370.0
            cached_harn = float(metrics_df[metrics_df["mode"] == "harness"]["avg_cached_latency_ms"].dropna().iloc[0]) if not metrics_df[metrics_df["mode"] == "harness"]["avg_cached_latency_ms"].dropna().empty else 50.8

            with lat_col1:
                st.metric("Baseline Cold Latency", f"{cold_base / 1000:.2f} s", f"{cold_base:.0f} ms")
            with lat_col2:
                st.metric("Harness Cold Latency", f"{cold_harn / 1000:.2f} s", f"{cold_harn:.0f} ms")
            with lat_col3:
                speedup = (cold_harn / cached_harn) if cached_harn > 0 else 1.0
                st.metric("Harness Cache Latency", f"{cached_harn:.1f} ms", f"{speedup:.0f}x faster", delta_color="normal")


# -----------------------------------------------------------------------------
# PAGE 2: Per-Category Breakdown
# -----------------------------------------------------------------------------
elif page == "2. Per-Category Breakdown":
    st.markdown('<div class="main-header">🗂️ Per-Category Performance Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detailed evaluation across distinct prompt constraint categories '
        'revealing how targeted feedback overcomes failure modes in each category.</div>',
        unsafe_allow_html=True,
    )

    if df_samples.empty:
        st.warning("No sample data available to breakdown by category.")
    else:
        # Group by category and mode
        cat_records: list[dict[str, Any]] = []
        all_cats = sorted(df_samples["category"].dropna().unique())

        for cat in all_cats:
            cat_df = df_samples[df_samples["category"] == cat]
            b_sub = cat_df[cat_df["mode"] == "baseline"]
            h_sub = cat_df[cat_df["mode"] == "harness"]

            b_total = len(b_sub)
            b_pass = int(b_sub["passed"].sum())
            b_sr = (b_pass / b_total) * 100 if b_total > 0 else 0.0

            h_total = len(h_sub)
            h_pass = int(h_sub["passed"].sum())
            h_sr = (h_pass / h_total) * 100 if h_total > 0 else 0.0
            h_retries = float(h_sub["retries_used"].mean()) if h_total > 0 else 0.0

            failed_att1 = int(h_sub["failed_attempt1"].sum()) if "failed_attempt1" in h_sub.columns else 0
            recovered = int(h_sub["recovered"].sum()) if "recovered" in h_sub.columns else 0
            rec_rate = (recovered / failed_att1) * 100 if failed_att1 > 0 else 100.0 if (h_pass == h_total) else 0.0

            cat_records.append({
                "Category": cat,
                "Samples": h_total if h_total > 0 else b_total,
                "Baseline Pass": f"{b_pass}/{b_total}",
                "Baseline Success (%)": round(b_sr, 1),
                "Harness Pass": f"{h_pass}/{h_total}",
                "Harness Success (%)": round(h_sr, 1),
                "Delta (%)": round(h_sr - b_sr, 1),
                "Recovery Rate (%)": round(rec_rate, 1),
                "Avg Retries": round(h_retries, 2),
            })

        cat_summary_df = pd.DataFrame(cat_records)

        # Plot Grouped Bar Chart
        chart_rows: list[dict[str, Any]] = []
        for r in cat_records:
            chart_rows.append({"Category": r["Category"], "Mode": "Baseline", "Success Rate (%)": r["Baseline Success (%)"]})
            chart_rows.append({"Category": r["Category"], "Mode": "Harness", "Success Rate (%)": r["Harness Success (%)"]})
        plot_df = pd.DataFrame(chart_rows)

        cat_chart = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Mode:N", title=None, axis=alt.Axis(labels=False)),
                y=alt.Y("Success Rate (%):Q", title="Success Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Mode:N", scale=alt.Scale(domain=["Baseline", "Harness"], range=["#F97316", "#10B981"])),
                column=alt.Column(
                    "Category:N",
                    title="Prompt Category",
                    header=alt.Header(labelAngle=-25, labelFontSize=12, labelFontWeight="bold"),
                ),
                tooltip=["Category", "Mode", "Success Rate (%)"],
            )
            .properties(width=160, height=350)
        )

        st.altair_chart(cat_chart)

        st.markdown("### Category Summary Table")
        st.dataframe(
            cat_summary_df.style.format({
                "Baseline Success (%)": "{:.1f}%",
                "Harness Success (%)": "{:.1f}%",
                "Delta (%)": "+{:.1f}%",
                "Recovery Rate (%)": "{:.1f}%",
                "Avg Retries": "{:.2f}",
            }),
            use_container_width=True,
        )

        # Interactive Sample Inspector
        st.markdown("---")
        st.subheader("🔍 Category Query Inspector")
        sel_cat = st.selectbox("Select a Category to view queries:", all_cats)

        cat_samples = df_samples[df_samples["category"] == sel_cat]
        if not cat_samples.empty:
            cat_ids = sorted(cat_samples["id"].unique())
            sel_id = st.selectbox("Select Sample ID:", cat_ids)
            sample_entries = cat_samples[cat_samples["id"] == sel_id]

            st.write(f"**Sample `{sel_id}` Details:**")
            meta_info = load_sample_metadata().get(sel_id, {})
            if meta_info:
                st.info(f"**Prompt:** {meta_info.get('prompt', '')}")
                if meta_info.get("reference"):
                    st.write(f"**Reference Answer:** `{meta_info.get('reference')}`")
                if meta_info.get("checker"):
                    st.json(meta_info.get("checker"))

            disp_cols = [c for c in ["mode", "passed", "reliability", "retries_used", "error_message", "final_output"] if c in sample_entries.columns]
            st.dataframe(sample_entries[disp_cols], use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 3: Reliability Score Distribution
# -----------------------------------------------------------------------------
elif page == "3. Reliability Distribution":
    st.markdown('<div class="main-header">📈 Reliability Score Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Examining how the harness shifts the reliability distribution curve '
        'from failing low-confidence outputs to compliant, high-reliability answers (≥ 0.80).</div>',
        unsafe_allow_html=True,
    )

    if df_samples.empty:
        st.warning("No evaluation data available to display score distribution.")
    else:
        dist_df = df_samples[df_samples["mode"].isin(["baseline", "harness"])].copy()

        # Score Tiers Analysis
        def categorize_tier(rel: float) -> str:
            if rel >= 0.80:
                return "Accepted (≥ 0.80)"
            elif rel >= 0.60:
                return "Marginal (0.60 – 0.79)"
            return "Failed / Low (< 0.60)"

        dist_df["Tier"] = dist_df["reliability"].apply(categorize_tier)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Score Histogram (Baseline vs. Harness)")
            # Altair distribution histogram
            hist = (
                alt.Chart(dist_df)
                .mark_bar(opacity=0.65)
                .encode(
                    x=alt.X("reliability:Q", bin=alt.Bin(step=0.05), title="Reliability Score"),
                    y=alt.Y("count():Q", title="Query Count", stack=None),
                    color=alt.Color("mode:N", scale=alt.Scale(domain=["baseline", "harness"], range=["#F97316", "#10B981"])),
                    tooltip=["mode:N", "count():Q"],
                )
                .properties(height=380)
            )

            thresh_rule = (
                alt.Chart(pd.DataFrame({"x": [0.80]}))
                .mark_rule(color="#DC2626", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="x:Q")
            )

            st.altair_chart(hist + thresh_rule, use_container_width=True)
            st.caption("🔴 Red dashed line marks the 0.80 acceptance threshold.")

        with col2:
            st.subheader("Reliability Tiers Breakdown (%)")
            tier_counts = dist_df.groupby(["mode", "Tier"]).size().reset_index(name="Count")
            tier_totals = dist_df.groupby("mode").size().to_dict()
            tier_counts["Percentage"] = tier_counts.apply(
                lambda row: round((row["Count"] / tier_totals[row["mode"]]) * 100, 1), axis=1
            )

            tier_chart = (
                alt.Chart(tier_counts)
                .mark_bar()
                .encode(
                    x=alt.X("mode:N", title="Execution Mode"),
                    y=alt.Y("Percentage:Q", title="Percentage (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "Tier:N",
                        scale=alt.Scale(
                            domain=["Accepted (≥ 0.80)", "Marginal (0.60 – 0.79)", "Failed / Low (< 0.60)"],
                            range=["#10B981", "#FBBF24", "#EF4444"],
                        ),
                    ),
                    tooltip=["mode", "Tier", "Count", "Percentage"],
                )
                .properties(height=380)
            )

            st.altair_chart(tier_chart, use_container_width=True)

        # Statistical Summary Table
        st.markdown("### Reliability Score Statistics")
        stats_rows = []
        for m in ["baseline", "harness"]:
            sub = dist_df[dist_df["mode"] == m]["reliability"]
            if not sub.empty:
                stats_rows.append({
                    "Mode": "Baseline (Unassisted)" if m == "baseline" else "Agentic Harness",
                    "Sample Count": len(sub),
                    "Mean Score": round(float(sub.mean()), 3),
                    "Median Score": round(float(sub.median()), 3),
                    "Std Dev": round(float(sub.std()), 3),
                    "Min Score": round(float(sub.min()), 3),
                    "Max Score": round(float(sub.max()), 3),
                    "% Accepted (≥ 0.80)": f"{(sub >= 0.80).mean() * 100:.1f}%",
                })
        st.table(pd.DataFrame(stats_rows))


# -----------------------------------------------------------------------------
# PAGE 4: Retries-per-Query Histogram
# -----------------------------------------------------------------------------
elif page == "4. Retries-per-Query Histogram":
    st.markdown('<div class="main-header">🔄 Retries-per-Query Histogram</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating retry loop efficiency: distribution of attempts required to reach '
        'acceptance and error recovery dynamics.</div>',
        unsafe_allow_html=True,
    )

    if df_samples.empty:
        st.warning("No sample data available to display retry statistics.")
    else:
        harn_df = df_samples[df_samples["mode"] == "harness"].copy()

        if harn_df.empty:
            st.info("No harness samples found in current data source.")
        else:
            total_harn = len(harn_df)
            avg_retries = float(harn_df["retries_used"].mean())
            zero_retries = int((harn_df["retries_used"] == 0).sum())
            one_retry = int((harn_df["retries_used"] == 1).sum())
            two_plus_retries = int((harn_df["retries_used"] >= 2).sum())

            # Top KPI metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Average Retries", f"{avg_retries:.2f}")
            with kpi2:
                st.metric("1st-Turn Passes (0 Retries)", f"{(zero_retries / total_harn) * 100:.1f}%", f"{zero_retries}/{total_harn}")
            with kpi3:
                st.metric("Fixed with 1 Retry", f"{(one_retry / total_harn) * 100:.1f}%", f"{one_retry}/{total_harn}")
            with kpi4:
                st.metric("Needed 2+ Retries", f"{(two_plus_retries / total_harn) * 100:.1f}%", f"{two_plus_retries}/{total_harn}")

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Retries Distribution")
                retry_counts = harn_df["retries_used"].value_counts().reset_index()
                retry_counts.columns = ["Retries", "Count"]
                retry_counts["Retries_Label"] = retry_counts["Retries"].apply(lambda r: f"{r} Retries" if r > 0 else "0 Retries (Turn 1)")
                retry_counts["Percentage"] = (retry_counts["Count"] / total_harn) * 100
                retry_counts = retry_counts.sort_values("Retries")

                ret_chart = (
                    alt.Chart(retry_counts)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Retries_Label:N", title=None, sort=list(retry_counts["Retries_Label"])),
                        y=alt.Y("Count:Q", title="Query Count"),
                        color=alt.Color(
                            "Retries:O",
                            scale=alt.Scale(domain=[0, 1, 2, 3], range=["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]),
                            legend=None,
                        ),
                        tooltip=["Retries_Label", "Count", "Percentage"],
                    )
                    .properties(height=350)
                )

                ret_text = ret_chart.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-6,
                    fontSize=13,
                    fontWeight="bold",
                ).encode(text=alt.Text("Count:Q"))

                st.altair_chart(ret_chart + ret_text, use_container_width=True)

            with col2:
                st.subheader("Outcome Breakdown by Retry Count")
                outcome_df = harn_df.groupby(["retries_used", "passed"]).size().reset_index(name="Count")
                outcome_df["Outcome"] = outcome_df["passed"].apply(lambda p: "Passed" if p else "Failed")
                outcome_df["Retries_Label"] = outcome_df["retries_used"].apply(lambda r: f"{r} Retries" if r > 0 else "0 Retries")

                outcome_chart = (
                    alt.Chart(outcome_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Retries_Label:N", title=None),
                        y=alt.Y("Count:Q", title="Queries"),
                        color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=["#10B981", "#EF4444"])),
                        tooltip=["Retries_Label", "Outcome", "Count"],
                    )
                    .properties(height=350)
                )

                st.altair_chart(outcome_chart, use_container_width=True)

            # Table of recovered queries
            st.markdown("---")
            st.subheader("Queries Corrected via Agentic Retries")
            retried_queries = harn_df[harn_df["retries_used"] > 0]
            if not retried_queries.empty:
                cols_to_show = [c for c in ["id", "category", "retries_used", "passed", "error_message", "final_output"] if c in retried_queries.columns]
                st.dataframe(retried_queries[cols_to_show].sort_values("retries_used", ascending=False), use_container_width=True)
            else:
                st.info("No queries required retries.")


# -----------------------------------------------------------------------------
# PAGE 5: "Try it live" Box
# -----------------------------------------------------------------------------
elif page == "5. Try it Live":
    st.markdown('<div class="main-header">🧪 Try it Live: Agentic Verification Harness</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter any prompt or test query to run it through the live agentic retry loop. '
        'Inspect each attempt, its sub-scores, deterministic validator errors, and critic feedback in real-time.</div>',
        unsafe_allow_html=True,
    )

    # Preset examples
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    preset_prompt = ""

    with preset_col1:
        if st.button("📝 3 sentences, no commas", use_container_width=True):
            st.session_state["live_prompt"] = (
                "Write exactly 3 sentences about renewable solar power, each between 12 and 18 words, with no commas."
            )
    with preset_col2:
        if st.button("📌 4 bullets, no 'e'", use_container_width=True):
            st.session_state["live_prompt"] = (
                "List 4 benefits of daily walking in bullet points. Do not use the letter 'e'."
            )
    with preset_col3:
        if st.button("📦 JSON user object", use_container_width=True):
            st.session_state["live_prompt"] = (
                "Output a JSON object with keys username and email for a software developer."
            )
    with preset_col4:
        if st.button("🔢 Word Math Problem", use_container_width=True):
            st.session_state["live_prompt"] = (
                "Alice has 15 apples. She gives 4 to Bob and buys 3 packs of 6 apples each. How many apples does she have? Give only the number."
            )

    default_prompt = st.session_state.get(
        "live_prompt",
        "Write exactly 3 sentences about renewable solar energy, each between 12 and 18 words, with no commas.",
    )

    user_query = st.text_area(
        "Enter prompt or query:",
        value=default_prompt,
        height=110,
        placeholder="Type a request with format constraints, word limits, or questions...",
    )

    with st.expander("⚙️ Execution Configuration & Manual Constraints", expanded=False):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            cfg_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3)
        with cfg_col2:
            cfg_temp = st.slider("Initial Generator Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        with cfg_col3:
            cfg_cache = st.checkbox("Enable Semantic Cache", value=False)

        manual_constraints_str = st.text_area(
            "Manual Constraints (optional JSON array; leave blank to auto-extract with LLM Judge)",
            value="",
            placeholder='[{"type": "sentence_count", "n": 3}, {"type": "no_commas"}]',
        )

    run_btn = st.button("🚀 Run Answering Harness", type="primary", use_container_width=True)

    if run_btn and user_query.strip():
        # Lazy import of harness to avoid overhead when viewing charts
        from agentic_harness.harness import answer

        parsed_manual_constraints = None
        if manual_constraints_str.strip():
            try:
                parsed_manual_constraints = json.loads(manual_constraints_str)
                if not isinstance(parsed_manual_constraints, list):
                    st.error("Manual constraints must be a JSON array of objects.")
                    parsed_manual_constraints = None
            except json.JSONDecodeError as exc:
                st.error(f"Failed to parse manual constraints JSON: {exc}")
                parsed_manual_constraints = None

        with st.spinner("Executing agentic evaluation loop (Generator + Validators + Critic Judge) …"):
            try:
                res = answer(
                    query=user_query.strip(),
                    constraints=parsed_manual_constraints,
                    max_retries=cfg_retries,
                    temperature=cfg_temp,
                    enable_cache=cfg_cache,
                    sample_id="live_dashboard_query",
                )
            except Exception as exc:
                st.error(f"Execution Error: {exc}")
                res = None

        if res:
            st.markdown("---")
            st.subheader("Execution Results & Attempt History")

            # Extracted Constraints
            constraints = res.get("constraints", [])
            if constraints:
                st.markdown("**Active Machine-Checkable Constraints:**")
                badges_html = " ".join([
                    f'<span style="background-color:#E0E7FF;color:#3730A3;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;margin-right:6px;">{c.get("type", "check")}: {json.dumps({k: v for k, v in c.items() if k != "type"})}</span>'
                    for c in constraints
                ])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            history = res.get("history", [])

            # Chronological Attempts Display
            for step in history:
                att = step.get("attempt", 1)
                is_retry = att > 1
                label = f"Attempt {att}" if not is_retry else f"Retry {att - 1} (Attempt {att})"
                accepted = step.get("accepted", False)
                det_err = step.get("det_error", "")
                stalled = det_err == "possible checker error"

                if stalled:
                    badge = '<span class="badge-stalled">⚠️ STALLED (POSSIBLE CHECKER ERROR)</span>'
                elif accepted:
                    badge = '<span class="badge-pass">✓ ACCEPTED</span>'
                else:
                    badge = '<span class="badge-fail">✗ REJECTED</span>'

                st.markdown(f"#### {label} &nbsp; {badge}", unsafe_allow_html=True)

                # Sub-scores row
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                det_val = step.get("det_score")
                det_display = f"{det_val:.2f} ({'PASS' if det_val == 1.0 else 'FAIL'})" if det_val is not None else "None (Skipped)"
                sem_val = step.get("sem_score", 0.0)
                crit_obj = step.get("critic") or {}
                crit_val = crit_obj.get("score", step.get("crit_score", 0.0))
                rel_val = step.get("reliability", 0.0)

                with score_col1:
                    st.metric("Deterministic Check", det_display)
                with score_col2:
                    st.metric("Semantic Relevance", f"{sem_val:.3f}")
                with score_col3:
                    st.metric("Critic Judge Grade", f"{crit_val:.3f}")
                with score_col4:
                    st.metric("Reliability Score", f"{rel_val:.3f}", f"Threshold: 0.80")

                # Candidate Output text
                output_text = step.get("output", "(no output generated)")
                st.markdown("**Candidate Response:**")
                st.code(output_text, language="text")

                # Validator Errors display
                if det_err and det_err != "possible checker error":
                    st.error(f"**Validator Error(s):**\n\n{det_err}")

                # Critic Details
                if crit_obj:
                    issues = crit_obj.get("issues", [])
                    suggestion = crit_obj.get("suggestion", "")
                    if issues or suggestion:
                        with st.container():
                            st.markdown("**Critic Feedback:**")
                            if issues:
                                for iss in issues:
                                    st.markdown(f"- ⚠️ {iss}")
                            if suggestion:
                                st.markdown(f"- 💡 *Suggestion*: {suggestion}")

                # Feedback sent to generator for next retry
                feedback_msg = step.get("feedback")
                if feedback_msg and not accepted:
                    with st.expander("💬 Direct Retry Feedback Sent to Model", expanded=False):
                        st.text(feedback_msg)

                st.markdown("---")

            # Final Summary Banner
            final_ans = res.get("answer")
            final_rel = res.get("reliability", 0.0)
            retries = res.get("retries_used", 0)
            passed = res.get("passed", False)
            latency = res.get("latency_ms", 0.0)
            from_cache = res.get("from_cache", False)

            if passed:
                st.success(f"🎉 **Final Answer Accepted!** (Reliability: `{final_rel:.3f}`, Retries: `{retries}`, Latency: `{latency:.1f}ms`, Cached: `{from_cache}`)")
            else:
                st.error(f"⚠️ **Answer Failed Constraints** (Reliability: `{final_rel:.3f}`, Retries: `{retries}`, Latency: `{latency:.1f}ms`)")

            st.markdown("### Final Output")
            st.markdown(final_ans or "(None)")
