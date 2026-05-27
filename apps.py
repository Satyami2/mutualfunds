"""
Mutual Fund Portfolio Dashboard
================================
Build a portfolio, see rolling returns vs Nifty 500, plus look-through
breakdowns: market cap, sector, and top stock holdings.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Page config + styles
# =============================================================================
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="centered",
)

st.markdown("""
<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 960px;}
    #MainMenu, footer, header {visibility: hidden;}

    h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
    .subtitle {opacity: 0.6; font-size: 0.95rem; margin-bottom: 2rem;}

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: rgba(128, 128, 128, 0.25) !important;
    }

    /* Buttons — theme-aware */
    .stButton button {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        background: transparent;
        font-weight: 500;
        transition: all 0.15s;
    }
    .stButton button:hover {
        border-color: rgba(128, 128, 128, 0.7);
        background: rgba(128, 128, 128, 0.08);
    }

    /* Section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin: 2rem 0 0.75rem 0;
    }

    /* Generic card */
    .card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 0.85rem;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 1.2rem;
    }
    .card-title {font-size: 1.05rem; font-weight: 600;}
    .card-meta  {font-size: 0.8rem;  opacity: 0.6;}

    /* Rolling-returns compare grid */
    .compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .compare-col {
        padding: 1rem 1.1rem;
        border-radius: 10px;
    }
    .compare-col.portfolio {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .compare-col.benchmark {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .compare-header {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.85rem;
    }
    .dot {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
    .dot.portfolio  {background: #3b82f6;}
    .dot.benchmark  {background: #9ca3af;}
    .dot.large      {background: #3b82f6;}
    .dot.mid        {background: #8b5cf6;}
    .dot.small      {background: #ec4899;}
    .dot.cash       {background: #9ca3af;}
    .dot.other      {background: #6b7280;}

    .compare-stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;}
    .compare-stat-label {
        font-size: 0.65rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em;
        opacity: 0.55; margin-bottom: 0.2rem;
    }
    .compare-stat-value {
        font-size: 1.25rem; font-weight: 700;
        letter-spacing: -0.02em; line-height: 1.1;
    }

    /* Verdict badge */
    .verdict {
        text-align: center; font-size: 0.85rem; font-weight: 600;
        padding: 0.65rem 1rem; border-radius: 8px; margin-top: 0.4rem;
    }
    .verdict.win  {background: rgba(16, 185, 129, 0.12); color: #10b981;
                   border: 1px solid rgba(16, 185, 129, 0.3);}
    .verdict.loss {background: rgba(239, 68, 68, 0.12); color: #ef4444;
                   border: 1px solid rgba(239, 68, 68, 0.3);}
    .verdict.tie  {background: rgba(128, 128, 128, 0.1); opacity: 0.75;
                   border: 1px solid rgba(128, 128, 128, 0.25);}

    /* Stacked-bar for cap allocation */
    .stack-bar {
        display: flex; height: 44px; border-radius: 10px; overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    .stack-seg {
        display: flex; align-items: center; justify-content: center;
        font-size: 0.82rem; font-weight: 600; color: white;
        transition: all 0.2s;
        min-width: 0;
    }
    .stack-legend {
        display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem;
        margin-top: 0.8rem;
    }
    .legend-row {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 0.85rem;
        padding: 0.25rem 0;
    }
    .legend-left {display: flex; align-items: center; gap: 0.55rem;}
    .legend-name {opacity: 0.85;}
    .legend-value {font-weight: 600; font-variant-numeric: tabular-nums;}

    /* Horizontal bar rows (sector, stocks) */
    .bar-row {
        display: grid;
        grid-template-columns: 1fr 60px;
        gap: 0.75rem;
        align-items: center;
        margin-bottom: 0.55rem;
    }
    .bar-label {
        font-size: 0.88rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bar-track {
        height: 22px;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 5px;
        position: relative;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s;
    }
    .bar-fill.blue   {background: linear-gradient(90deg, #3b82f6, #60a5fa);}
    .bar-fill.purple {background: linear-gradient(90deg, #8b5cf6, #a78bfa);}
    .bar-fill.pink   {background: linear-gradient(90deg, #ec4899, #f472b6);}
    .bar-fill.gray   {background: linear-gradient(90deg, #6b7280, #9ca3af);}
    .bar-pct {
        font-size: 0.85rem;
        font-weight: 600;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    /* Stock row */
    .stock-row {
        display: grid;
        grid-template-columns: 28px 1fr auto 70px;
        gap: 0.75rem;
        align-items: center;
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        margin-bottom: 0.45rem;
    }
    .stock-rank {
        font-size: 0.8rem;
        font-weight: 600;
        opacity: 0.4;
        font-variant-numeric: tabular-nums;
    }
    .stock-name {
        font-size: 0.92rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .cap-badge {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
    }
    .cap-badge.large  {background: rgba(59, 130, 246, 0.15); color: #3b82f6;}
    .cap-badge.mid    {background: rgba(139, 92, 246, 0.15); color: #8b5cf6;}
    .cap-badge.small  {background: rgba(236, 72, 153, 0.15); color: #ec4899;}
    .stock-pct {
        font-size: 0.95rem;
        font-weight: 700;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .pos {color: #10b981;}
    .neg {color: #ef4444;}
    .neu {color: inherit;}

    /* Reduce gap on view-selector radio */
    div[role="radiogroup"] {gap: 0.4rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# File paths
# =============================================================================
DATA_DIR = Path(__file__).parent

NAV_FILES = {
    "Large Cap": "largecap.parquet",
    "Flexi Cap": "flexicap.parquet",
    "Multi Cap": "multicap.parquet",
    "Mid Cap":   "midcap.parquet",
    "Small Cap": "smallcap.parquet",
}
BENCHMARK_FILE = "nifty500.parquet"
ASSETTYPE_FILE = "assettype_allocations.parquet"
SECTOR_FILE    = "sector_allocation.parquet"
STOCKS_FILE    = "stock_allocations.parquet"
AMFI_FILE      = "AMFI.parquet"


# =============================================================================
# Data loaders (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_category(path: str, category: str) -> pd.DataFrame:
    """Load a pre-parsed NAV parquet file."""
    data = pd.read_parquet(path)
    data.columns = pd.MultiIndex.from_product([[category], data.columns])
    return data


@st.cache_data(show_spinner=False)
def load_benchmark(path: str) -> pd.Series:
    """Load pre-parsed Nifty 500 parquet."""
    return pd.read_parquet(path)["Close"]


@st.cache_data(show_spinner=False)
def load_holdings():
    """Load pre-parsed holdings parquets.
    Returns (asset_df, sector_df, stock_df, missing_files)."""
    missing = []
    empty_at = pd.DataFrame(columns=["Scheme", "AssetType", "Allocation"])
    empty_sec = pd.DataFrame(columns=["Scheme", "Sector", "Allocation"])
    empty_stk = pd.DataFrame(columns=["Scheme", "Stock", "Sector", "Allocation"])

    p = DATA_DIR / ASSETTYPE_FILE
    if p.exists():
        at = pd.read_parquet(p)
    else:
        at = empty_at
        missing.append(ASSETTYPE_FILE)

    p = DATA_DIR / SECTOR_FILE
    if p.exists():
        sec = pd.read_parquet(p)
    else:
        sec = empty_sec
        missing.append(SECTOR_FILE)

    p = DATA_DIR / STOCKS_FILE
    if p.exists():
        stk = pd.read_parquet(p)
    else:
        stk = empty_stk
        missing.append(STOCKS_FILE)

    return at, sec, stk, missing


@st.cache_data(show_spinner=False)
def load_amfi_classification():
    """Returns (dict, missing_flag). Maps normalized stock name -> 'Large Cap' / 'Mid Cap'.
    Stocks not in dict are classified as Small Cap downstream."""
    p = DATA_DIR / AMFI_FILE
    if not p.exists():
        return {}, True
    amfi = pd.read_parquet(p)
    return {_normalize_stock(n): cat for n, cat in zip(amfi["Company Name"], amfi["Category"])}, False


def _normalize_stock(name) -> str:
    """Normalize stock names for matching across files."""
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\bltd\.?\b|\blimited\b", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner=False)
def load_all():
    frames, catalogue = [], {}
    for cat, fname in NAV_FILES.items():
        p = DATA_DIR / fname
        if not p.exists():
            st.error(
                f"Missing required NAV file: `{fname}`. "
                f"Run `python convert_to_parquet.py` locally to generate it from "
                f"your .xlsx files, then push the .parquet files to your repo."
            )
            st.stop()
        df = load_category(str(p), cat)
        frames.append(df)
        catalogue[cat] = [c for _, c in df.columns]
    nav = pd.concat(frames, axis=1, sort=False).sort_index().ffill(limit=5)

    bench_path = DATA_DIR / BENCHMARK_FILE
    if not bench_path.exists():
        st.error(f"Missing required benchmark file: {BENCHMARK_FILE}. Add it to your repo.")
        st.stop()
    bench = load_benchmark(str(bench_path))

    asset_df, sector_df, stock_df, missing_holdings = load_holdings()
    amfi_map, amfi_missing = load_amfi_classification()
    if amfi_missing:
        missing_holdings.append(AMFI_FILE)
    return nav, catalogue, bench, asset_df, sector_df, stock_df, amfi_map, missing_holdings


# =============================================================================
# Computations
# =============================================================================
def build_portfolio(nav: pd.DataFrame, selections: list) -> pd.Series:
    cols = [(cat, f) for cat, f, _ in selections]
    weights = np.array([w for _, _, w in selections]) / 100.0
    sub = nav[cols].dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)
    normalized = sub.divide(sub.iloc[0])
    return pd.Series(normalized.values @ weights, index=sub.index)


def rolling_cagr(series: pd.Series, window_days: int) -> pd.Series:
    if len(series) < window_days + 1:
        return pd.Series(dtype=float)
    years = window_days / 365.25
    return ((series / series.shift(window_days)) ** (1 / years) - 1).dropna()


def look_through(df: pd.DataFrame, selections: list, group_col: str) -> pd.Series:
    """
    Weight a holdings dataframe by the user's fund weights.
    df has columns: Scheme, <group_col>, Allocation.
    Returns a series indexed by group_col with total weighted % of portfolio.
    """
    if df.empty:
        return pd.Series(dtype=float)
    total_w = sum(w for _, _, w in selections)
    if total_w <= 0:
        return pd.Series(dtype=float)

    parts = []
    for _, fund, w in selections:
        share = w / total_w
        fund_rows = df[df["Scheme"] == fund]
        if fund_rows.empty:
            continue
        contrib = fund_rows.groupby(group_col)["Allocation"].sum() * share
        parts.append(contrib)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).groupby(level=0).sum().sort_values(ascending=False)


def classify_cap(stock: str, amfi_map: dict) -> str:
    return amfi_map.get(_normalize_stock(stock), "Small Cap")


def covered_funds(funds: list, df: pd.DataFrame) -> tuple[list, list]:
    """Returns (covered, missing) from selected funds based on holdings data."""
    schemes_in_data = set(df["Scheme"].unique())
    covered = [f for f in funds if f in schemes_in_data]
    missing = [f for f in funds if f not in schemes_in_data]
    return covered, missing


# =============================================================================
# Load data
# =============================================================================
with st.spinner("Loading fund data..."):
    nav, catalogue, benchmark, asset_df, sector_df, stock_df, amfi_map, missing_holdings = load_all()


def pick_oldest(cat: str) -> str:
    funds = catalogue[cat]
    best, best_d = funds[0], nav[(cat, funds[0])].first_valid_index() or pd.Timestamp.max
    for f in funds[1:]:
        d = nav[(cat, f)].first_valid_index()
        if d is not None and d < best_d:
            best, best_d = f, d
    return best


# =============================================================================
# Header
# =============================================================================
st.markdown("# Portfolio Dashboard")
st.markdown(
    '<div class="subtitle">Build a portfolio. Explore returns, market-cap mix, '
    'sectors, and top stock holdings.</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Fund selection (always visible at top)
# =============================================================================
if "selections" not in st.session_state:
    st.session_state.selections = [
        {"category": "Large Cap", "fund": pick_oldest("Large Cap"), "weight": 40.0},
        {"category": "Mid Cap",   "fund": pick_oldest("Mid Cap"),   "weight": 30.0},
        {"category": "Small Cap", "fund": pick_oldest("Small Cap"), "weight": 30.0},
    ]

st.markdown('<div class="section-label">Your portfolio</div>', unsafe_allow_html=True)

to_delete = None
for i, row in enumerate(st.session_state.selections):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 4, 1.3, 0.5])
        with c1:
            cat = st.selectbox(
                "Category", list(catalogue.keys()),
                index=list(catalogue.keys()).index(row["category"]),
                key=f"cat_{i}", label_visibility="collapsed",
            )
            if cat != row["category"]:
                row["category"] = cat
                row["fund"] = pick_oldest(cat)
        with c2:
            funds = catalogue[cat]
            inception = {f: nav[(cat, f)].first_valid_index() for f in funds}

            def fmt(f, _im=inception):
                d = _im.get(f)
                return f"{f}  ·  since {d.year}" if d is not None else f
            idx = funds.index(row["fund"]) if row["fund"] in funds else 0
            row["fund"] = st.selectbox(
                "Fund", funds, index=idx, format_func=fmt,
                key=f"fund_{i}", label_visibility="collapsed",
            )
        with c3:
            row["weight"] = st.number_input(
                "Weight", min_value=0.0, max_value=100.0,
                value=float(row["weight"]), step=5.0,
                key=f"w_{i}", label_visibility="collapsed",
            )
        with c4:
            st.markdown('<div style="padding-top:4px"></div>', unsafe_allow_html=True)
            if st.button("✕", key=f"del_{i}", help="Remove"):
                to_delete = i

if to_delete is not None:
    st.session_state.selections.pop(to_delete)
    st.rerun()

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    if st.button("＋ Add fund", use_container_width=True):
        st.session_state.selections.append(
            {"category": "Large Cap", "fund": pick_oldest("Large Cap"), "weight": 0.0}
        )
        st.rerun()
with col_b:
    if st.button("⚖ Equal weights", use_container_width=True):
        n = len(st.session_state.selections)
        if n:
            for r in st.session_state.selections:
                r["weight"] = round(100.0 / n, 2)
        st.rerun()
with col_c:
    total = sum(r["weight"] for r in st.session_state.selections)
    color = "#10b981" if abs(total - 100) < 0.01 else "#ef4444"
    st.markdown(
        f'<div style="text-align:right; padding-top:8px; color:{color}; '
        f'font-weight:600;">Total: {total:.0f}%</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Build selections list + early exits
# =============================================================================
selections = [
    (r["category"], r["fund"], r["weight"])
    for r in st.session_state.selections if r["weight"] > 0
]
if not selections:
    st.info("Add at least one fund with a non-zero weight to see results.")
    st.stop()

if abs(total - 100) > 0.01:
    st.warning(f"Weights sum to {total:.0f}%, not 100%. Showing proportional weights.")


# =============================================================================
# View selector — holdings views are disabled if their data file is missing
# =============================================================================
st.markdown('<div class="section-label">What do you want to see?</div>', unsafe_allow_html=True)

views_all = ["📈 Rolling Returns", "🥧 Market Cap", "🏭 Sectors", "🏢 Top Stocks"]

# Decide which views are unavailable
disabled_reasons = {}
if stock_df.empty:
    disabled_reasons["🥧 Market Cap"] = STOCKS_FILE
    disabled_reasons["🏢 Top Stocks"] = STOCKS_FILE
if sector_df.empty:
    disabled_reasons["🏭 Sectors"] = SECTOR_FILE

views_available = [v for v in views_all if v not in disabled_reasons]

if missing_holdings:
    files_str = ", ".join(f"`{f}`" for f in missing_holdings)
    st.info(
        f"ℹ️ Holdings views are disabled. To enable them, add these file(s) to "
        f"your repo: {files_str}"
    )

view = st.radio(
    "View", views_available,
    horizontal=True, label_visibility="collapsed",
)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# =============================================================================
# VIEW 1: Rolling Returns
# =============================================================================
def color_for(v):
    if v > 0.10:  return "pos"
    if v < 0:     return "neg"
    return "neu"


def stat_html(label, value, klass="neu"):
    return (
        f'<div><div class="compare-stat-label">{label}</div>'
        f'<div class="compare-stat-value {klass}">{value}</div></div>'
    )


def column_html(name, dot_class, stats):
    inner = "".join(stat_html(l, v, c) for l, v, c in stats)
    return (
        f'<div class="compare-col {dot_class}">'
        f'  <div class="compare-header">'
        f'    <span class="dot {dot_class}"></span>{name}'
        f'  </div>'
        f'  <div class="compare-stats">{inner}</div>'
        f'</div>'
    )


def render_rolling_returns():
    portfolio = build_portfolio(nav, selections)
    if portfolio.empty or len(portfolio) < 252:
        st.error(
            "Not enough overlapping data for these funds. "
            "Try replacing newer funds with ones showing an older `since` year."
        )
        return

    fund_inceptions = [(f, nav[(cat, f)].first_valid_index()) for cat, f, _ in selections]
    youngest = max(fund_inceptions, key=lambda x: x[1] or pd.Timestamp.min)
    backtest_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    if backtest_years < 5:
        st.info(
            f"⚠️  Your backtest is limited to **{backtest_years:.1f} years** "
            f"because **{youngest[0]}** only has data since "
            f"**{youngest[1].strftime('%b %Y')}**. Swap it for an older fund "
            f"to unlock 3Y / 5Y rolling returns."
        )

    bench = benchmark.loc[portfolio.index[0]:portfolio.index[-1]].copy()
    bench = bench.reindex(portfolio.index).ffill()

    windows = [("1-Year Returns", 252), ("3-Year Returns", 252 * 3), ("5-Year Returns", 252 * 5)]
    period_start = portfolio.index[0].strftime("%b %Y")
    period_end   = portfolio.index[-1].strftime("%b %Y")
    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        f'Based on data from {period_start} to {period_end}</div>',
        unsafe_allow_html=True,
    )

    for title, days in windows:
        rr_p = rolling_cagr(portfolio, days)
        rr_b = rolling_cagr(bench, days)

        if rr_p.empty:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">'
                f'<div class="card-title">{title}</div>'
                f'<div class="card-meta">Not enough history</div>'
                f'</div>'
                f'<div style="opacity:0.7; font-size:0.9rem;">'
                f'Need at least {days // 252} years of overlapping data. '
                f'Limiting fund: <b>{youngest[0]}</b> '
                f'(since {youngest[1].strftime("%b %Y")}).'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            continue

        pmn, pmd, pmx = rr_p.min(), rr_p.median(), rr_p.max()
        bmn, bmd, bmx = rr_b.min(), rr_b.median(), rr_b.max()
        pos_pct = (rr_p > 0).mean() * 100

        portfolio_col = column_html("Your Portfolio", "portfolio", [
            ("Min", f"{pmn*100:.1f}%", color_for(pmn)),
            ("Median", f"{pmd*100:.1f}%", color_for(pmd)),
            ("Max", f"{pmx*100:.1f}%", color_for(pmx)),
        ])
        benchmark_col = column_html("Nifty 500", "benchmark", [
            ("Min", f"{bmn*100:.1f}%", color_for(bmn)),
            ("Median", f"{bmd*100:.1f}%", color_for(bmd)),
            ("Max", f"{bmx*100:.1f}%", color_for(bmx)),
        ])

        diff = pmd - bmd
        if abs(diff) < 0.005:
            v_cls, v_txt = "tie",  f"In line with Nifty 500 ({diff*100:+.1f}% on median)"
        elif diff > 0:
            v_cls, v_txt = "win",  f"↑ Portfolio beat Nifty 500 by {diff*100:.1f}% on median"
        else:
            v_cls, v_txt = "loss", f"↓ Portfolio lagged Nifty 500 by {abs(diff)*100:.1f}% on median"

        st.markdown(
            f'<div class="card">'
            f'  <div class="card-header">'
            f'    <div class="card-title">{title}</div>'
            f'    <div class="card-meta">{len(rr_p):,} windows · portfolio positive in {pos_pct:.0f}%</div>'
            f'  </div>'
            f'  <div class="compare-grid">{portfolio_col}{benchmark_col}</div>'
            f'  <div class="verdict {v_cls}">{v_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# VIEW 2: Market Cap Allocation
# =============================================================================
def render_market_cap():
    selected_funds = [f for _, f, _ in selections]
    covered, missing = covered_funds(selected_funds, stock_df)

    if not covered:
        st.error("None of your selected funds have holdings data available.")
        return
    if missing:
        st.info(f"ℹ️ Holdings data unavailable for {len(missing)} fund(s): "
                f"_{', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}_. "
                "Their weight is excluded from this breakdown.")

    # Look-through stock holdings, classify each, sum by cap
    weighted_rows = []
    total_w_covered = sum(w for _, f, w in selections if f in covered)
    if total_w_covered <= 0:
        st.error("No weight allocated to funds with holdings data.")
        return

    for _, fund, w in selections:
        if fund not in covered:
            continue
        share = w / total_w_covered
        fund_rows = stock_df[stock_df["Scheme"] == fund].copy()
        if fund_rows.empty:
            continue
        fund_rows["Cap"] = fund_rows["Stock"].apply(lambda s: classify_cap(s, amfi_map))
        cap_alloc = fund_rows.groupby("Cap")["Allocation"].sum() * share
        weighted_rows.append(cap_alloc)

    cap_totals = pd.concat(weighted_rows).groupby(level=0).sum() if weighted_rows else pd.Series(dtype=float)

    # Also compute "non-equity" (cash + debt + etc.) from asset type data
    non_equity = 0.0
    at_covered, _ = covered_funds(selected_funds, asset_df)
    total_w_at = sum(w for _, f, w in selections if f in at_covered)
    if total_w_at > 0:
        for _, fund, w in selections:
            if fund not in at_covered:
                continue
            share = w / total_w_at
            fund_at = asset_df[asset_df["Scheme"] == fund]
            non_eq = fund_at[~fund_at["AssetType"].str.contains("Equit", case=False, na=False)]
            non_equity += non_eq["Allocation"].sum() * share

    # Normalize so segments sum to ~100
    large = cap_totals.get("Large Cap", 0.0)
    mid   = cap_totals.get("Mid Cap",   0.0)
    small = cap_totals.get("Small Cap", 0.0)
    cash  = non_equity
    total = large + mid + small + cash
    if total <= 0:
        st.error("Could not compute allocation breakdown.")
        return

    # Normalize each segment to a true % of 100 (the holdings sum may be slightly
    # off due to disclosed-vs-undisclosed positions, so we rescale)
    segs = [
        ("Large Cap", large, "#3b82f6", "large"),
        ("Mid Cap",   mid,   "#8b5cf6", "mid"),
        ("Small Cap", small, "#ec4899", "small"),
        ("Cash / Debt / Other", cash, "#9ca3af", "cash"),
    ]
    scale = 100.0 / total
    segs = [(n, v * scale, c, cls) for n, v, c, cls in segs]

    st.markdown(
        '<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        'Look-through analysis as of April 2026. Stocks not in AMFI top 250 are classified as Small Cap.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Stacked bar
    bar_segs = "".join(
        f'<div class="stack-seg" style="background:{c}; width:{v:.4f}%;" '
        f'title="{n}: {v:.1f}%">{v:.0f}%</div>'
        for n, v, c, _ in segs if v > 0
    )
    legend = "".join(
        f'<div class="legend-row">'
        f'  <div class="legend-left">'
        f'    <span class="dot {cls}"></span>'
        f'    <span class="legend-name">{n}</span>'
        f'  </div>'
        f'  <div class="legend-value">{v:.1f}%</div>'
        f'</div>'
        for n, v, _, cls in segs
    )

    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Where your money sits</div>'
        f'    <div class="card-meta">By market cap</div>'
        f'  </div>'
        f'  <div class="stack-bar">{bar_segs}</div>'
        f'  <div class="stack-legend">{legend}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# VIEW 3: Sector Allocation
# =============================================================================
def render_sectors():
    selected_funds = [f for _, f, _ in selections]
    covered, missing = covered_funds(selected_funds, sector_df)

    if not covered:
        st.error("None of your selected funds have sector data available.")
        return
    if missing:
        st.info(f"ℹ️ Sector data unavailable for {len(missing)} fund(s): "
                f"_{', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}_. "
                "Their weight is excluded.")

    total_w = sum(w for _, f, w in selections if f in covered)
    if total_w <= 0:
        st.error("No weight allocated to funds with sector data.")
        return

    # Look-through
    parts = []
    for _, fund, w in selections:
        if fund not in covered:
            continue
        share = w / total_w
        f_rows = sector_df[sector_df["Scheme"] == fund]
        parts.append(f_rows.groupby("Sector")["Allocation"].sum() * share)
    sector_totals = pd.concat(parts).groupby(level=0).sum().sort_values(ascending=False)

    show_n = st.slider("How many sectors to show?", min_value=5, max_value=20,
                       value=10, step=1, key="sector_n")

    top = sector_totals.head(show_n)
    others = sector_totals.iloc[show_n:].sum()
    max_val = top.max() if not top.empty else 1.0

    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        f'Look-through analysis as of April 2026. Top {show_n} of {len(sector_totals)} sectors.'
        f'</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for sector, value in top.items():
        width_pct = (value / max_val) * 100
        rows_html += (
            f'<div class="bar-row">'
            f'  <div>'
            f'    <div class="bar-label">{sector}</div>'
            f'    <div class="bar-track">'
            f'      <div class="bar-fill blue" style="width:{width_pct:.2f}%"></div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="bar-pct">{value:.2f}%</div>'
            f'</div>'
        )
    if others > 0:
        width_pct = (others / max_val) * 100
        rows_html += (
            f'<div class="bar-row" style="opacity:0.6;">'
            f'  <div>'
            f'    <div class="bar-label">Others ({len(sector_totals) - show_n} sectors)</div>'
            f'    <div class="bar-track">'
            f'      <div class="bar-fill gray" style="width:{width_pct:.2f}%"></div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="bar-pct">{others:.2f}%</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Sector exposure</div>'
        f'    <div class="card-meta">Top {show_n} sectors</div>'
        f'  </div>'
        f'  {rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# VIEW 4: Top Stocks
# =============================================================================
def render_top_stocks():
    selected_funds = [f for _, f, _ in selections]
    covered, missing = covered_funds(selected_funds, stock_df)

    if not covered:
        st.error("None of your selected funds have stock-holdings data available.")
        return
    if missing:
        st.info(f"ℹ️ Stock data unavailable for {len(missing)} fund(s): "
                f"_{', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}_. "
                "Their weight is excluded.")

    total_w = sum(w for _, f, w in selections if f in covered)
    if total_w <= 0:
        st.error("No weight allocated to funds with stock data.")
        return

    parts = []
    for _, fund, w in selections:
        if fund not in covered:
            continue
        share = w / total_w
        f_rows = stock_df[stock_df["Scheme"] == fund]
        parts.append(f_rows.groupby("Stock")["Allocation"].sum() * share)
    stock_totals = pd.concat(parts).groupby(level=0).sum().sort_values(ascending=False)

    show_n = st.slider("How many stocks to show?", min_value=5, max_value=25,
                       value=10, step=1, key="stocks_n")

    top = stock_totals.head(show_n)

    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        f'Look-through analysis as of April 2026. Top {show_n} of {len(stock_totals):,} unique stocks.'
        f'</div>',
        unsafe_allow_html=True,
    )

    cap_color = {"Large Cap": "large", "Mid Cap": "mid", "Small Cap": "small"}
    cap_short = {"Large Cap": "Large", "Mid Cap": "Mid",  "Small Cap": "Small"}

    rows_html = ""
    for i, (stock, value) in enumerate(top.items(), start=1):
        cap = classify_cap(stock, amfi_map)
        cls = cap_color[cap]
        rows_html += (
            f'<div class="stock-row">'
            f'  <div class="stock-rank">{i:02d}</div>'
            f'  <div class="stock-name">{stock}</div>'
            f'  <div class="cap-badge {cls}">{cap_short[cap]}</div>'
            f'  <div class="stock-pct">{value:.2f}%</div>'
            f'</div>'
        )

    top_sum = top.sum()
    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Top {show_n} stock holdings</div>'
        f'    <div class="card-meta">{top_sum:.1f}% of portfolio</div>'
        f'  </div>'
        f'  {rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Route to the chosen view
# =============================================================================
if view.endswith("Rolling Returns"):
    render_rolling_returns()
elif view.endswith("Market Cap"):
    render_market_cap()
elif view.endswith("Sectors"):
    render_sectors()
elif view.endswith("Top Stocks"):
    render_top_stocks()


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    '<div style="text-align:center; opacity:0.4; font-size:0.75rem; margin-top:2.5rem;">'
    'Past performance is not indicative of future results.'
    '</div>',
    unsafe_allow_html=True,
)
