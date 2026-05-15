"""
Mutual Fund Portfolio Backtesting Dashboard
============================================
Build a custom portfolio from Large Cap / Mid Cap / Small Cap mutual funds,
allocate weights, and see what your portfolio would have returned historically.

Run with:
    streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mutual Fund Portfolio Backtester",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent  # expects xlsx files next to app.py

FILES = {
    "Large Cap": "largecap.xlsx",
    "Mid Cap":   "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
}


@st.cache_data(show_spinner="Loading NAV data...")
def load_category(path: str, category: str) -> pd.DataFrame:
    """Parse one of the ACE MF NAV exports into a tidy NAV dataframe."""
    raw = pd.read_excel(path, header=None)

    # Row index 2 has fund names (column 0 is blank — that's the Date column)
    fund_names = raw.iloc[2, 1:].tolist()
    # Drop columns whose header is NaN (trailing empty cols)
    keep_mask = [pd.notna(f) for f in fund_names]
    fund_names = [f for f, k in zip(fund_names, keep_mask) if k]

    # Data starts at row index 4
    data = raw.iloc[4:, : len(fund_names) + 1].copy()
    data.columns = ["Date"] + fund_names

    # Date column: coerce, drop disclaimer / junk rows
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).set_index("Date").sort_index()

    # NAV columns: coerce to numeric
    for c in fund_names:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Tag category in a multi-index style by renaming columns later
    data.columns = pd.MultiIndex.from_product([[category], data.columns])
    return data


@st.cache_data(show_spinner="Loading NAV data...")
def load_all() -> tuple[pd.DataFrame, dict]:
    """Load all three category files and stitch into one wide NAV frame."""
    frames = []
    catalogue = {}  # category -> list of fund names
    for cat, fname in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            st.error(f"Missing data file: {path}")
            st.stop()
        df = load_category(str(path), cat)
        frames.append(df)
        catalogue[cat] = [c for _, c in df.columns]

    nav = pd.concat(frames, axis=1).sort_index()
    # Forward-fill small NAV gaps (holidays / missing days) — capped to 5 days
    nav = nav.ffill(limit=5)
    return nav, catalogue


# ---------------------------------------------------------------------------
# Portfolio math
# ---------------------------------------------------------------------------
def build_portfolio_nav(
    nav: pd.DataFrame,
    selections: list[tuple[str, str, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    selections: list of (category, fund_name, weight_pct)
    Returns a dataframe with each fund's normalized growth (₹1 invested) and
    the blended portfolio value column.
    """
    cols = [(cat, f) for cat, f, _ in selections]
    weights = np.array([w for _, _, w in selections]) / 100.0

    sub = nav.loc[start:end, cols].copy()
    # Drop rows where ANY selected fund is NaN — keeps backtest fair
    sub = sub.dropna(how="any")
    if sub.empty:
        return sub

    # Normalize each fund to start at 1
    normalized = sub.divide(sub.iloc[0])
    # Portfolio = weighted sum of normalized fund values
    portfolio = normalized.values @ weights
    normalized.columns = [f"{f}" for _, f in normalized.columns]
    normalized["Portfolio"] = portfolio
    return normalized


def cagr(series: pd.Series) -> float:
    """Compound annual growth rate from a normalized series (starts at 1)."""
    if len(series) < 2:
        return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return series.iloc[-1] ** (1 / years) - 1


def rolling_returns(series: pd.Series, window_days: int) -> pd.Series:
    """Annualized rolling returns over a fixed window."""
    if len(series) < window_days + 1:
        return pd.Series(dtype=float)
    years = window_days / 365.25
    ratio = series / series.shift(window_days)
    return ratio ** (1 / years) - 1


def drawdown(series: pd.Series) -> pd.Series:
    return series / series.cummax() - 1


# ---------------------------------------------------------------------------
# Sidebar — fund selection
# ---------------------------------------------------------------------------
nav, catalogue = load_all()

st.sidebar.title("🎛️ Build Your Portfolio")
st.sidebar.caption(
    "Pick funds from any category, then set what % of your money goes into each."
)

def _pick_oldest(cat: str) -> str:
    """Pick fund with the longest history in a category for sensible defaults."""
    funds = catalogue[cat]
    best = funds[0]
    best_date = nav[(cat, best)].first_valid_index() or pd.Timestamp.max
    for f in funds[1:]:
        d = nav[(cat, f)].first_valid_index()
        if d is not None and d < best_date:
            best, best_date = f, d
    return best

if "selections" not in st.session_state:
    # Seed with funds that have the longest history in each category
    st.session_state.selections = [
        {"category": "Large Cap", "fund": _pick_oldest("Large Cap"), "weight": 40.0},
        {"category": "Mid Cap",   "fund": _pick_oldest("Mid Cap"),   "weight": 30.0},
        {"category": "Small Cap", "fund": _pick_oldest("Small Cap"), "weight": 30.0},
    ]

# Render each row
to_delete = None
for i, row in enumerate(st.session_state.selections):
    with st.sidebar.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            cat = st.selectbox(
                "Category",
                list(catalogue.keys()),
                index=list(catalogue.keys()).index(row["category"]),
                key=f"cat_{i}",
            )
            # If category changed, reset fund to first in that category
            if cat != row["category"]:
                row["category"] = cat
                row["fund"] = catalogue[cat][0]

            fund_options = catalogue[cat]
            # Show inception date next to fund name so user picks wisely
            inception_map = {
                f: nav[(cat, f)].first_valid_index() for f in fund_options
            }
            def _label(f, _cat=cat, _im=inception_map):
                d = _im.get(f)
                tag = d.strftime("%Y") if d is not None else "—"
                return f"{f}  ·  since {tag}"

            fund_idx = fund_options.index(row["fund"]) if row["fund"] in fund_options else 0
            row["fund"] = st.selectbox(
                "Fund",
                fund_options,
                index=fund_idx,
                format_func=_label,
                key=f"fund_{i}",
            )
            row["weight"] = st.number_input(
                "Allocation %",
                min_value=0.0, max_value=100.0,
                value=float(row["weight"]),
                step=5.0,
                key=f"w_{i}",
            )
        with cols[1]:
            st.write("")
            st.write("")
            if st.button("🗑️", key=f"del_{i}", help="Remove this fund"):
                to_delete = i

if to_delete is not None:
    st.session_state.selections.pop(to_delete)
    st.rerun()

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("➕ Add fund", use_container_width=True):
        st.session_state.selections.append(
            {"category": "Large Cap", "fund": catalogue["Large Cap"][0], "weight": 0.0}
        )
        st.rerun()
with c2:
    if st.button("⚖️ Equal weight", use_container_width=True):
        n = len(st.session_state.selections)
        if n:
            for r in st.session_state.selections:
                r["weight"] = round(100.0 / n, 2)
        st.rerun()

# Weight check
total_weight = sum(r["weight"] for r in st.session_state.selections)
st.sidebar.markdown(f"**Total allocation: {total_weight:.1f}%**")
if abs(total_weight - 100.0) > 0.01:
    st.sidebar.warning("⚠️ Weights should sum to 100%.")

st.sidebar.divider()
investment_amount = st.sidebar.number_input(
    "Lump-sum investment (₹)", min_value=1000.0, value=100000.0, step=10000.0
)

# Date range
min_date = nav.index.min().date()
max_date = nav.index.max().date()
st.sidebar.divider()
st.sidebar.subheader("📅 Backtest period")
date_range = st.sidebar.date_input(
    "From – To",
    value=(max(min_date, pd.Timestamp("2018-01-01").date()), max_date),
    min_value=min_date, max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📈 Mutual Fund Portfolio Backtester")
st.caption(
    "Select funds, set weights, and see how your blended portfolio would have "
    "performed over history — total return, CAGR, drawdowns, and rolling returns."
)

selections = [
    (r["category"], r["fund"], r["weight"])
    for r in st.session_state.selections
    if r["weight"] > 0
]

if not selections:
    st.info("👈 Add at least one fund with a non-zero weight in the sidebar.")
    st.stop()

if abs(total_weight - 100.0) > 0.01:
    st.warning(
        f"Your weights sum to **{total_weight:.1f}%**, not 100%. "
        "Results are still computed but interpret them as proportional weights."
    )

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

port = build_portfolio_nav(nav, selections, start_ts, end_ts)

if port.empty:
    st.error(
        "No overlapping data for the selected funds in this date range. "
        "Some funds may have launched later. Try a more recent start date or "
        "pick funds with longer history."
    )
    # Show inception dates to help user
    incept = []
    for cat, f, _ in selections:
        s = nav[(cat, f)].dropna()
        if not s.empty:
            incept.append({"Fund": f, "Earliest data": s.index.min().date()})
    if incept:
        st.dataframe(pd.DataFrame(incept), use_container_width=True, hide_index=True)
    st.stop()

# ----- Headline metrics ------------------------------------------------------
portfolio_series = port["Portfolio"]
final_value = investment_amount * portfolio_series.iloc[-1]
total_return = portfolio_series.iloc[-1] - 1
ann_return = cagr(portfolio_series)
years = (portfolio_series.index[-1] - portfolio_series.index[0]).days / 365.25

# Annualized volatility from daily returns
daily_ret = portfolio_series.pct_change().dropna()
ann_vol = daily_ret.std() * np.sqrt(252)
sharpe = (ann_return - 0.06) / ann_vol if ann_vol > 0 else np.nan  # 6% RFR proxy
max_dd = drawdown(portfolio_series).min()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Final value", f"₹{final_value:,.0f}",
          f"{total_return*100:+.1f}% total")
m2.metric("CAGR", f"{ann_return*100:.2f}%",
          help="Compound Annual Growth Rate")
m3.metric("Volatility", f"{ann_vol*100:.2f}%",
          help="Annualized standard deviation of daily returns")
m4.metric("Max drawdown", f"{max_dd*100:.2f}%",
          help="Worst peak-to-trough decline")
m5.metric("Sharpe ratio", f"{sharpe:.2f}",
          help="(CAGR − 6% RFR) / annualized volatility")

st.caption(
    f"Period: **{portfolio_series.index[0].date()}** → "
    f"**{portfolio_series.index[-1].date()}**  ·  "
    f"**{years:.2f} years**  ·  "
    f"₹{investment_amount:,.0f} grew to **₹{final_value:,.0f}**"
)

# ----- Tabs ------------------------------------------------------------------
tab_growth, tab_rolling, tab_dd, tab_individual, tab_table = st.tabs(
    ["📈 Growth", "🔄 Rolling Returns", "📉 Drawdown", "🔍 Fund Comparison", "📋 Data"]
)

# Growth chart
with tab_growth:
    df_plot = (port * investment_amount).reset_index().melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )
    fig = px.line(
        df_plot, x="Date", y="Value", color="Series",
        title=f"Growth of ₹{investment_amount:,.0f}",
        labels={"Value": "Portfolio value (₹)"},
    )
    fig.update_traces(line=dict(width=1.5))
    # Make the portfolio line bold
    for tr in fig.data:
        if tr.name == "Portfolio":
            tr.line.width = 3
            tr.line.color = "#111"
    fig.update_layout(hovermode="x unified", height=500, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

# Rolling returns
with tab_rolling:
    st.markdown(
        "Rolling annualized returns over fixed windows. Each point shows the "
        "CAGR you'd have earned over the previous N years if you'd invested then. "
        "The summary stats answer: **what's the historical range of outcomes?**"
    )
    windows = {"1Y": 252, "3Y": 252 * 3, "5Y": 252 * 5}
    available = {k: v for k, v in windows.items() if len(portfolio_series) > v + 1}

    if not available:
        st.warning("Backtest period too short for any rolling window. Need at least 1 year.")
    else:
        # Summary table
        summary = []
        for label, w in available.items():
            r = rolling_returns(portfolio_series, w).dropna()
            if r.empty:
                continue
            summary.append({
                "Window": label,
                "Min": f"{r.min()*100:.2f}%",
                "25th %ile": f"{r.quantile(0.25)*100:.2f}%",
                "Median": f"{r.median()*100:.2f}%",
                "Mean": f"{r.mean()*100:.2f}%",
                "75th %ile": f"{r.quantile(0.75)*100:.2f}%",
                "Max": f"{r.max()*100:.2f}%",
                "% positive": f"{(r > 0).mean()*100:.1f}%",
                "Observations": len(r),
            })
        st.subheader("Historical range of annualized returns")
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        # Chart
        chosen = st.selectbox(
            "Rolling-window chart",
            list(available.keys()),
            index=min(1, len(available) - 1),
        )
        r = rolling_returns(portfolio_series, available[chosen]).dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name=f"{chosen} rolling CAGR",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=r.median(), line_dash="dot", line_color="green",
                      annotation_text=f"Median: {r.median():.1f}%")
        fig.update_layout(
            title=f"{chosen} rolling annualized return",
            yaxis_title="Annualized return (%)",
            height=450, hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Histogram
        fig2 = px.histogram(r, nbins=40, title=f"Distribution of {chosen} rolling returns")
        fig2.update_layout(height=350, xaxis_title="Annualized return (%)",
                           showlegend=False, yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)

# Drawdown
with tab_dd:
    dd = drawdown(portfolio_series) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                             fillcolor="rgba(220,50,50,0.3)",
                             line=dict(color="rgb(180,30,30)", width=1.5),
                             name="Drawdown"))
    fig.update_layout(
        title="Portfolio drawdown (peak-to-trough decline)",
        yaxis_title="Drawdown (%)",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    worst_dd_date = dd.idxmin()
    st.info(
        f"📉 **Worst drawdown:** {dd.min():.2f}% on **{worst_dd_date.date()}** — "
        "this is how much your portfolio would have been down from its prior peak "
        "at the worst moment in this period."
    )

# Individual fund comparison
with tab_individual:
    st.markdown("How each selected fund performed individually (₹1 normalized).")
    cmp = []
    for col in port.columns:
        s = port[col]
        cmp.append({
            "Name": col,
            "Final (× initial)": f"{s.iloc[-1]:.2f}x",
            "Total return": f"{(s.iloc[-1]-1)*100:.2f}%",
            "CAGR": f"{cagr(s)*100:.2f}%",
            "Volatility": f"{s.pct_change().std()*np.sqrt(252)*100:.2f}%",
            "Max DD": f"{drawdown(s).min()*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(cmp), use_container_width=True, hide_index=True)

    # Composition pie
    comp_df = pd.DataFrame(
        [{"Fund": f, "Category": c, "Weight": w} for c, f, w in selections]
    )
    fig = px.pie(comp_df, values="Weight", names="Fund",
                 title="Portfolio composition", hole=0.4)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Raw data
with tab_table:
    st.markdown("Daily portfolio value (₹) and individual fund growth (× initial).")
    display_df = port.copy()
    display_df["Portfolio (₹)"] = display_df["Portfolio"] * investment_amount
    st.dataframe(display_df.round(4), use_container_width=True)

    # CSV download
    csv = display_df.to_csv().encode()
    st.download_button(
        "⬇️ Download as CSV", csv,
        file_name="portfolio_backtest.csv", mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "**Disclaimer:** Past performance is not indicative of future results. "
    "This tool is for educational/illustrative purposes only and does not "
    "constitute investment advice. NAV data sourced from ACE MF exports."
)
