"""
JSE Stock Screening App – Rebuilt
iPad-friendly · Full multi-indicator filter · Portfolio tracker
Deploy on Streamlit Community Cloud → Add to Home Screen
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JSE Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# iPad-friendly CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 16.5px !important; }
    .stButton > button {
        height: 3.1rem !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-size: 1.05rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3.3rem;
        font-size: 1.1rem !important;
        font-weight: 600;
        padding: 0 1.3rem !important;
    }
    .block-container { padding-top: 1.1rem !important; padding-bottom: 1.8rem !important; }
    h1 { font-size: 1.85rem !important; margin-bottom: 0.25rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filter constants (exactly as specified)
# ---------------------------------------------------------------------------
PRICE_MIN = 10.0
PRICE_MAX = 120.0
MIN_AVG_VOLUME = 10_000

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_tickers(path: str = "jse_active_tickers.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            return pd.DataFrame()
        return df.drop_duplicates(subset=["ticker"])
    except Exception:
        return pd.DataFrame({
            "ticker": ["ABG.JO", "FSR.JO", "SBK.JO", "NPN.JO", "MTN.JO", "SHP.JO",
                       "SOL.JO", "ANG.JO", "GFI.JO", "IMP.JO", "CPI.JO", "NED.JO",
                       "VOD.JO", "DSY.JO", "GND.JO", "REM.JO", "SLM.JO", "OUT.JO"],
            "name": ["Absa", "FirstRand", "Standard Bank", "Naspers", "MTN",
                     "Shoprite", "Sasol", "AngloGold", "Gold Fields", "Impala",
                     "Capitec", "Nedbank", "Vodacom", "Discovery", "Grindrod",
                     "Remgro", "Sanlam", "OUTsurance"]
        })


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch 4-hour OHLCV data (better for swing trading). Prices converted to ZAR."""
    try:
        t = yf.Ticker(ticker)
        # 4-hour bars – yfinance typically allows ~60-730 days depending on ticker
        df = t.history(period=period, interval="4h", auto_adjust=True)
        if df.empty or len(df) < 60:
            # Fallback to daily if 4h fails
            df = t.history(period="1y", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 60:
                return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        # Yahoo JSE prices are in ZAc → convert to ZAR
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 100.0
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Rolling Moving Average (used by RSI and Alligator)."""
    return series.ewm(alpha=1/length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pure pandas/numpy indicators – no pandas-ta (avoids numba/Python 3.14 issues)."""
    if df.empty or len(df) < 60:
        return df
    out = df.copy()

    # --- RSI(14) ---
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, 14)
    avg_loss = _rma(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # --- CCI(20) ---
    tp = (out["high"] + out["low"] + out["close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # --- MACD (12, 26, 9) ---
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- Ichimoku ---
    high_9 = out["high"].rolling(9).max()
    low_9 = out["low"].rolling(9).min()
    out["tenkan"] = (high_9 + low_9) / 2

    high_26 = out["high"].rolling(26).max()
    low_26 = out["low"].rolling(26).min()
    out["kijun"] = (high_26 + low_26) / 2

    out["span_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(26)

    high_52 = out["high"].rolling(52).max()
    low_52 = out["low"].rolling(52).min()
    out["span_b"] = ((high_52 + low_52) / 2).shift(26)

    # --- Alligator (Bill Williams) ---
    median = (out["high"] + out["low"]) / 2
    out["jaw"] = _rma(median, 13).shift(8)
    out["teeth"] = _rma(median, 8).shift(5)
    out["lips"] = _rma(median, 5).shift(3)

    return out



def passes_filters(df: pd.DataFrame) -> tuple[bool, dict]:
    """All conditions must be true."""
    if df.empty or len(df) < 30:
        return False, {}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    vol_avg_20 = df["volume"].tail(20).mean()
    if pd.isna(vol_avg_20) or vol_avg_20 < MIN_AVG_VOLUME:
        return False, {"reason": "volume"}

    close = float(last["close"])
    if not (PRICE_MIN <= close <= PRICE_MAX):
        return False, {"reason": "price", "close": close}

    metrics = {
        "close": round(close, 2),
        "vol_avg_20": int(vol_avg_20),
        "rsi": round(float(last.get("rsi", np.nan)), 1) if pd.notna(last.get("rsi")) else None,
        "cci": round(float(last.get("cci", np.nan)), 1) if pd.notna(last.get("cci")) else None,
    }

    # RSI 50–75
    rsi = last.get("rsi", np.nan)
    if pd.isna(rsi) or not (50 <= rsi <= 75):
        return False, {**metrics, "reason": "rsi"}

    # CCI > 0
    cci = last.get("cci", np.nan)
    if pd.isna(cci) or cci <= 0:
        return False, {**metrics, "reason": "cci"}

    # MACD > Signal
    macd = last.get("macd", np.nan)
    signal = last.get("macd_signal", np.nan)
    if pd.isna(macd) or pd.isna(signal) or macd <= signal:
        return False, {**metrics, "reason": "macd"}

    # Ichimoku: price above cloud + Tenkan > Kijun
    span_a = last.get("span_a", np.nan)
    span_b = last.get("span_b", np.nan)
    tenkan = last.get("tenkan", np.nan)
    kijun = last.get("kijun", np.nan)
    if any(pd.isna(x) for x in [span_a, span_b, tenkan, kijun]):
        return False, {**metrics, "reason": "ichimoku_missing"}
    if close <= max(span_a, span_b) or tenkan <= kijun:
        return False, {**metrics, "reason": "ichimoku"}

    # Alligator: Lips > Teeth > Jaw and rising
    lips = last.get("lips", np.nan)
    teeth = last.get("teeth", np.nan)
    jaw = last.get("jaw", np.nan)
    if any(pd.isna(x) for x in [lips, teeth, jaw]):
        return False, {**metrics, "reason": "alligator_missing"}
    if not (lips > teeth > jaw):
        return False, {**metrics, "reason": "alligator_order"}
    if not (lips > prev.get("lips", lips) and teeth > prev.get("teeth", teeth) and jaw > prev.get("jaw", jaw)):
        return False, {**metrics, "reason": "alligator_rising"}

    metrics["all_ok"] = True
    return True, metrics


@st.cache_data(ttl=1800, show_spinner="Scanning tickers… first run can take 1–3 minutes")
def scan_all_tickers(ticker_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in ticker_df.iterrows():
        ticker = str(row["ticker"]).strip()
        name = row.get("name", ticker)
        df = fetch_ohlcv(ticker)
        if df.empty:
            continue
        df = compute_indicators(df)
        ok, metrics = passes_filters(df)
        if ok:
            results.append({
                "Ticker": ticker.replace(".JO", ""),
                "Name": name,
                "Close": metrics["close"],
                "Vol20": metrics["vol_avg_20"],
                "RSI": metrics["rsi"],
                "CCI": metrics["cci"],
                "OK": "✅",
                "yf_ticker": ticker,
            })
    return pd.DataFrame(results)


def plot_review_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        subplot_titles=(title, "MACD", "RSI & CCI", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    if "span_a" in df.columns and "span_b" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["span_a"], name="Span A",
                                 line=dict(width=1.2, color="rgba(33,150,243,0.7)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["span_b"], name="Span B",
                                 line=dict(width=1.2, color="rgba(255,152,0,0.7)"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.12)"), row=1, col=1)
    if "tenkan" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["tenkan"], name="Tenkan",
                                 line=dict(width=1.5, color="#2196F3")), row=1, col=1)
    if "kijun" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["kijun"], name="Kijun",
                                 line=dict(width=1.5, color="#FF9800")), row=1, col=1)

    for col, color, nm in [("lips", "#4CAF50", "Lips"), ("teeth", "#FF5722", "Teeth"), ("jaw", "#3F51B5", "Jaw")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm,
                                     line=dict(width=1.3, color=color)), row=1, col=1)

    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                 line=dict(color="#2196F3", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                 line=dict(color="#FF9800", width=1.5)), row=2, col=1)
        if "macd_hist" in df.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Hist",
                                 marker_color=colors, opacity=0.55), row=2, col=1)

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                                 line=dict(color="#9C27B0", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    if "cci" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cci"], name="CCI",
                                 line=dict(color="#00BCD4", width=1.3)), row=3, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Vol",
                         marker_color="rgba(100,100,100,0.35)"), row=4, col=1)

    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        margin=dict(l=30, r=15, t=50, b=15),
        template="plotly_white",
        font=dict(size=13),
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------
def init_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=[
            "Ticker", "Name", "Qty", "Avg Price", "Notes", "Date Added"
        ])


def save_portfolio_csv() -> bytes:
    buf = io.BytesIO()
    st.session_state.portfolio.to_csv(buf, index=False)
    return buf.getvalue()


def load_portfolio_from_csv(uploaded):
    try:
        df = pd.read_csv(uploaded)
        if not {"Ticker", "Qty", "Avg Price"}.issubset(set(df.columns)):
            st.error("CSV must contain at least: Ticker, Qty, Avg Price")
            return
        st.session_state.portfolio = df
        st.success("Portfolio loaded")
    except Exception as e:
        st.error(f"Load failed: {e}")


def add_or_update_position(ticker: str, name: str, qty: float, price: float, notes: str = ""):
    port = st.session_state.portfolio
    mask = port["Ticker"] == ticker
    if mask.any():
        idx = port[mask].index[0]
        old_qty = float(port.at[idx, "Qty"])
        old_avg = float(port.at[idx, "Avg Price"])
        new_qty = old_qty + qty
        new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty
        port.at[idx, "Qty"] = new_qty
        port.at[idx, "Avg Price"] = round(new_avg, 4)
        if notes:
            port.at[idx, "Notes"] = notes
    else:
        new_row = pd.DataFrame([{
            "Ticker": ticker,
            "Name": name,
            "Qty": qty,
            "Avg Price": price,
            "Notes": notes,
            "Date Added": datetime.now().strftime("%Y-%m-%d")
        }])
        st.session_state.portfolio = pd.concat([port, new_row], ignore_index=True)


def sell_position(ticker: str, qty: float):
    port = st.session_state.portfolio
    mask = port["Ticker"] == ticker
    if not mask.any():
        st.warning("Ticker not in portfolio")
        return
    idx = port[mask].index[0]
    current = float(port.at[idx, "Qty"])
    if qty >= current:
        st.session_state.portfolio = port.drop(idx).reset_index(drop=True)
        st.success(f"Fully closed {ticker}")
    else:
        port.at[idx, "Qty"] = current - qty
        st.success(f"Sold {qty}. Remaining: {current - qty}")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    init_portfolio()

    # Header
    c1, c2 = st.columns([5, 1])
    with c1:
        st.title("📈 JSE Stock Screener")
        st.caption("4-Hour timeframe · Liquidity + Ichimoku + Alligator + MACD + RSI + CCI · iPad ready")
    with c2:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tab_rec, tab_review, tab_port = st.tabs([
        "📋 Recommendations",
        "🔍 Review Chart",
        "💼 Portfolio"
    ])

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    with tab_rec:
        st.markdown("#### Stocks that pass **every** filter (4-Hour bars)")
        st.markdown(
            f"**Timeframe: 4-Hour** · Vol ≥ {MIN_AVG_VOLUME:,} · Price R{PRICE_MIN}–R{PRICE_MAX} · "
            "Ichimoku (above cloud + Tenkan>Kijun) · Alligator (Lips>Teeth>Jaw rising) · "
            "MACD > Signal · RSI 50–75 · CCI > 0"
        )

        ticker_df = load_tickers()

        if st.button("▶ Run Full Scan", type="primary", use_container_width=True):
            results = scan_all_tickers(ticker_df)
            st.session_state["recs"] = results

        if "recs" in st.session_state:
            recs = st.session_state["recs"]
            if recs.empty:
                st.warning("No tickers currently pass all filters.")
            else:
                st.success(f"**{len(recs)}** recommendations found")
                display = recs[["Ticker", "Name", "Close", "Vol20", "RSI", "CCI", "OK"]].copy()
                display.columns = ["Ticker", "Name", "Close (R)", "Avg Vol 20", "RSI", "CCI", ""]
                st.dataframe(display, use_container_width=True, hide_index=True, height=400)

                st.session_state["rec_tickers"] = recs["Ticker"].tolist()
                st.session_state["rec_map"] = dict(zip(recs["Ticker"], recs["yf_ticker"]))
                st.session_state["name_map"] = dict(zip(recs["Ticker"], recs["Name"]))
        else:
            st.info("Tap **Run Full Scan** to start.")

    # ------------------------------------------------------------------
    # Review Chart
    # ------------------------------------------------------------------
    with tab_review:
        if "rec_tickers" not in st.session_state or not st.session_state["rec_tickers"]:
            st.warning("Run a scan in the Recommendations tab first.")
        else:
            selected = st.selectbox("Select stock to review", options=st.session_state["rec_tickers"])
            yf_sym = st.session_state["rec_map"].get(selected, selected + ".JO")
            name = st.session_state["name_map"].get(selected, selected)

            with st.spinner("Loading chart…"):
                df = fetch_ohlcv(yf_sym)
                if df.empty:
                    st.error("Could not load data")
                else:
                    df = compute_indicators(df)
                    fig = plot_review_chart(df, f"{selected} · {name}")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                    st.markdown("### ✅ Accept & Add to Portfolio")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        qty = st.number_input("Quantity", min_value=1.0, value=100.0, step=10.0)
                    with c2:
                        price_paid = st.number_input(
                            "Price paid (ZAR)", min_value=0.01,
                            value=float(round(df["close"].iloc[-1], 2)),
                            step=0.05, format="%.2f"
                        )
                    with c3:
                        notes = st.text_input("Notes", placeholder="optional")

                    if st.button("➕ Add / Increase Position", type="primary", use_container_width=True):
                        add_or_update_position(selected, name, qty, price_paid, notes)
                        st.success(f"Added {qty:.0f} × {selected} @ R{price_paid:.2f}")
                        st.balloons()

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    with tab_port:
        c_down, c_up = st.columns(2)
        with c_down:
            st.download_button(
                "⬇️ Download portfolio.csv",
                data=save_portfolio_csv(),
                file_name="portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_up:
            up = st.file_uploader("Upload portfolio.csv", type=["csv"], label_visibility="collapsed")
            if up:
                load_portfolio_from_csv(up)

        st.markdown("---")
        port = st.session_state.portfolio

        if port.empty:
            st.info("Portfolio is empty. Accept a recommendation from the Review tab.")
        else:
            enriched = []
            total_pl = 0.0
            for _, row in port.iterrows():
                tkr = row["Ticker"]
                yf_t = tkr if str(tkr).endswith(".JO") else str(tkr) + ".JO"
                df = fetch_ohlcv(yf_t, period="3mo")
                last_price = np.nan
                pl = np.nan
                pl_pct = np.nan
                signal = ""
                tip = ""
                if not df.empty:
                    df = compute_indicators(df)
                    last_price = float(df["close"].iloc[-1])
                    cost = float(row["Qty"]) * float(row["Avg Price"])
                    value = float(row["Qty"]) * last_price
                    pl = value - cost
                    pl_pct = (pl / cost * 100) if cost else 0
                    total_pl += pl

                    last = df.iloc[-1]
                    rsi = last.get("rsi", 50)
                    macd_v = last.get("macd", 0)
                    sig_v = last.get("macd_signal", 0)
                    if pd.notna(rsi) and rsi > 75:
                        signal = "⚠️ RSI high"
                        tip = "Consider taking profit"
                    elif pd.notna(macd_v) and pd.notna(sig_v) and macd_v < sig_v:
                        signal = "⚠️ MACD ↓"
                        tip = "Momentum weakening"
                    elif last.get("lips", 0) < last.get("teeth", 0):
                        signal = "⚠️ Alligator turn"
                        tip = "Trend may be ending"
                    else:
                        signal = "✅ Hold"
                        tip = "Signals still good"

                enriched.append({
                    "Ticker": tkr,
                    "Name": row.get("Name", ""),
                    "Qty": int(row["Qty"]),
                    "Avg": round(float(row["Avg Price"]), 2),
                    "Last": round(last_price, 2) if not pd.isna(last_price) else "–",
                    "P/L R": round(pl, 0) if not pd.isna(pl) else "–",
                    "P/L %": f"{pl_pct:.1f}%" if not pd.isna(pl_pct) else "–",
                    "Signal": signal,
                    "Tip": tip,
                })

            edf = pd.DataFrame(enriched)
            st.dataframe(edf, use_container_width=True, hide_index=True, height=360)
            st.metric("Total Unrealised P/L", f"R {total_pl:,.0f}")

            st.markdown("### 🔻 Sell")
            sell_tkr = st.selectbox("Ticker to sell", options=port["Ticker"].tolist())
            max_q = float(port.loc[port["Ticker"] == sell_tkr, "Qty"].iloc[0])
            sell_qty = st.number_input("Quantity to sell", min_value=1.0, max_value=max_q,
                                       value=min(100.0, max_q), step=1.0)
            if st.button("Sell", type="secondary", use_container_width=True):
                sell_position(sell_tkr, sell_qty)
                st.rerun()

        st.caption("💡 Download portfolio.csv regularly — Streamlit Cloud restarts clear session data.")


if __name__ == "__main__":
    main()
