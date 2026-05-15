"""
Mutual Fund Portfolio — Rolling Returns
========================================
Pick funds, set weights, see historical rolling returns. That's it.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config + styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Rolling Returns",
    page_icon="📈",
    layout="centered",
)

st.markdown("""
<style>
    /* Tighten layout */
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 960px;}

    /* Hide streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Headline */
    h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
    .subtitle {opacity: 0.6; font-size: 0.95rem; margin-bottom: 2rem;}

    /* Selection cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: #e5e7eb !important;
    }

    /* Buttons — theme-aware, inherit text color from Streamlit */
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
    .stButton button p {
        font-weight: 500;
    }

    /* Section headers */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin: 2rem 0 0.75rem 0;
    }

    /* Rolling return cards — theme-aware */
    .rr-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 0.85rem;
    }
    .rr-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 1.2rem;
    }
    .rr-title {font-size: 1.05rem; font-weight: 600;}
    .rr-meta  {font-size: 0.8rem;  opacity: 0.6;}

    /* Comparison grid: portfolio | nifty 500 */
    .compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .compare-col {
        padding: 1rem 1.1rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.06);
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
    .dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
    }
    .dot.portfolio  {background: #3b82f6;}
    .dot.benchmark  {background: #9ca3af;}

    .compare-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }
    .compare-stat-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.55;
        margin-bottom: 0.2rem;
    }
    .compare-stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    /* Verdict badge — portfolio beat / lagged benchmark on median */
    .verdict {
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        margin-top: 0.4rem;
    }
    .verdict.win {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .verdict.loss {
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .verdict.tie {
        background: rgba(128, 128, 128, 0.1);
        opacity: 0.75;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .pos {color: #10b981;}
    .neg {color: #ef4444;}
    .neu {color: inherit;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

FILES = {
    "Large Cap": "largecap.xlsx",
    "Mid Cap":   "midcap.xlsx",
    "Small Cap": "smallcap.xlsx",
}

BENCHMARK_FILE = "nifty500.xlsx"


@st.cache_data(show_spinner=False)
def load_category(path: str, category: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    fund_names = raw.iloc[2, 1:].tolist()
    keep = [pd.notna(f) for f in fund_names]
    fund_names = [f for f, k in zip(fund_names, keep) if k]

    data = raw.iloc[4:, : len(fund_names) + 1].copy()
    data.columns = ["Date"] + fund_names
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in fund_names:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data.columns = pd.MultiIndex.from_product([[category], data.columns])
    return data


@st.cache_data(show_spinner=False)
def load_benchmark(path: str) -> pd.Series:
    """Load Nifty 500 close prices as a single time series."""
    raw = pd.read_excel(path, header=None)
    data = raw.iloc[3:, [1, 2]].copy()
    data.columns = ["Date", "Close"]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    return data.dropna().set_index("Date").sort_index()["Close"]


@st.cache_data(show_spinner=False)
def load_all() -> tuple[pd.DataFrame, dict, pd.Series]:
    frames, catalogue = [], {}
    for cat, fname in FILES.items():
        p = DATA_DIR / fname
        if not p.exists():
            st.error(f"Missing data file: {p}"); st.stop()
        df = load_category(str(p), cat)
        frames.append(df)
        catalogue[cat] = [c for _, c in df.columns]
    nav = pd.concat(frames, axis=1).sort_index().ffill(limit=5)

    bench_path = DATA_DIR / BENCHMARK_FILE
    if not bench_path.exists():
        st.error(f"Missing benchmark file: {bench_path}"); st.stop()
    benchmark = load_benchmark(str(bench_path))

    return nav, catalogue, benchmark


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def build_portfolio(nav, selections):
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
    ratio = series / series.shift(window_days)
    return (ratio ** (1 / years) - 1).dropna()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading fund data..."):
    nav, catalogue, benchmark = load_all()


def pick_oldest(cat: str) -> str:
    funds = catalogue[cat]
    best, best_d = funds[0], nav[(cat, funds[0])].first_valid_index() or pd.Timestamp.max
    for f in funds[1:]:
        d = nav[(cat, f)].first_valid_index()
        if d is not None and d < best_d:
            best, best_d = f, d
    return best


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Portfolio Rolling Returns")
st.markdown(
    '<div class="subtitle">Build a portfolio and see how it would have performed '
    'over every 1, 3, and 5-year window — compared against the Nifty 500.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fund selection
# ---------------------------------------------------------------------------
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
    color = "#059669" if abs(total - 100) < 0.01 else "#dc2626"
    st.markdown(
        f'<div style="text-align:right; padding-top:8px; color:{color}; '
        f'font-weight:600;">Total: {total:.0f}%</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Compute & display
# ---------------------------------------------------------------------------
selections = [
    (r["category"], r["fund"], r["weight"])
    for r in st.session_state.selections if r["weight"] > 0
]

if not selections:
    st.info("Add at least one fund with a non-zero weight to see results.")
    st.stop()

if abs(total - 100) > 0.01:
    st.warning(f"Weights sum to {total:.0f}%, not 100%. Results shown as proportional weights.")

portfolio = build_portfolio(nav, selections)

if portfolio.empty or len(portfolio) < 252:
    st.error(
        "Not enough overlapping data for these funds. "
        "Some funds may be newer — try replacing them with funds showing an older `since` year."
    )
    st.stop()

# Align benchmark to the same period as the portfolio for apples-to-apples
bench = benchmark.loc[portfolio.index[0]:portfolio.index[-1]].copy()
bench = bench.reindex(portfolio.index).ffill()

# Rolling returns
windows = [
    ("1-Year Returns", 252),
    ("3-Year Returns", 252 * 3),
    ("5-Year Returns", 252 * 5),
]

st.markdown('<div class="section-label">Portfolio vs Nifty 500</div>', unsafe_allow_html=True)

period_start = portfolio.index[0].strftime("%b %Y")
period_end   = portfolio.index[-1].strftime("%b %Y")
st.markdown(
    f'<div style="color:#9ca3af; font-size:0.85rem; margin-bottom:1rem;">'
    f'Based on data from {period_start} to {period_end}'
    f'</div>',
    unsafe_allow_html=True,
)


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
    """stats = [(label, value_str, color_class), ...]"""
    inner = "".join(stat_html(l, v, c) for l, v, c in stats)
    return (
        f'<div class="compare-col {dot_class}">'
        f'  <div class="compare-header">'
        f'    <span class="dot {dot_class}"></span>{name}'
        f'  </div>'
        f'  <div class="compare-stats">{inner}</div>'
        f'</div>'
    )


for title, days in windows:
    rr_p = rolling_cagr(portfolio, days)
    rr_b = rolling_cagr(bench,     days)

    if rr_p.empty:
        st.markdown(
            f'<div class="rr-card">'
            f'<div class="rr-header">'
            f'<div class="rr-title">{title}</div>'
            f'<div class="rr-meta">Not enough history</div>'
            f'</div>'
            f'<div style="color:#9ca3af; font-size:0.9rem;">'
            f'Need at least {days // 252} years of overlapping fund data. '
            f'Try picking funds with longer history.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        continue

    pmn, pmd, pmx = rr_p.min(), rr_p.median(), rr_p.max()
    bmn, bmd, bmx = rr_b.min(), rr_b.median(), rr_b.max()
    pos_pct = (rr_p > 0).mean() * 100

    portfolio_col = column_html(
        "Your Portfolio", "portfolio",
        [
            ("Min",    f"{pmn*100:.1f}%", color_for(pmn)),
            ("Median", f"{pmd*100:.1f}%", color_for(pmd)),
            ("Max",    f"{pmx*100:.1f}%", color_for(pmx)),
        ],
    )

    benchmark_col = column_html(
        "Nifty 500", "benchmark",
        [
            ("Min",    f"{bmn*100:.1f}%", color_for(bmn)),
            ("Median", f"{bmd*100:.1f}%", color_for(bmd)),
            ("Max",    f"{bmx*100:.1f}%", color_for(bmx)),
        ],
    )

    # Verdict: did portfolio beat benchmark on median?
    diff = pmd - bmd
    if abs(diff) < 0.005:
        verdict_class = "tie"
        verdict_text = f"In line with Nifty 500 ({diff*100:+.1f}% on median)"
    elif diff > 0:
        verdict_class = "win"
        verdict_text = f"↑ Portfolio beat Nifty 500 by {diff*100:.1f}% on median"
    else:
        verdict_class = "loss"
        verdict_text = f"↓ Portfolio lagged Nifty 500 by {abs(diff)*100:.1f}% on median"

    st.markdown(
        f'<div class="rr-card">'
        f'  <div class="rr-header">'
        f'    <div class="rr-title">{title}</div>'
        f'    <div class="rr-meta">{len(rr_p):,} windows · portfolio positive in {pos_pct:.0f}%</div>'
        f'  </div>'
        f'  <div class="compare-grid">'
        f'    {portfolio_col}'
        f'    {benchmark_col}'
        f'  </div>'
        f'  <div class="verdict {verdict_class}">{verdict_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center; color:#9ca3af; font-size:0.75rem; margin-top:2rem;">'
    'Past performance is not indicative of future results.'
    '</div>',
    unsafe_allow_html=True,
)
