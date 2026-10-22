"""
VolNet-X: NIFTY 50 realized-volatility forecasting dashboard.
HAR vs HAR + India VIX vs a from-scratch NumPy LSTM, scored out-of-sample.

Reads two files produced by the notebook (copy them into ./data):
    lstm_direct_test_forecasts.csv   (log-RV forecasts for every test day)
    lstm_direct_metrics.csv          (final scoreboard table)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="VolNet-X | NIFTY 50 volatility forecasting",
                   page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FORECASTS_CSV = DATA_DIR / "lstm_direct_test_forecasts.csv"
METRICS_CSV = DATA_DIR / "lstm_direct_metrics.csv"

# display name -> column in the forecasts file
MODELS = {
    "HAR": "har",
    "HAR + VIX": "har_vix_linear",
    "LSTM (3-seed avg)": "lstm_direct_avg",
}
# display name -> row label in the metrics file
METRIC_ROWS = {
    "Naive (yesterday)": "naive (yesterday)",
    "HAR": "HAR",
    "HAR + VIX": "HAR + VIX (linear)",
    "LSTM (3-seed avg)": "LSTM direct (avg seeds)",
    "50/50 HAR + LSTM": "50/50 HAR + LSTM",
}
COLORS = {
    "Actual": "rgba(128,128,128,0.85)",
    "HAR": "#1f77b4",
    "HAR + VIX": "#2ca02c",
    "LSTM (3-seed avg)": "#ff7f0e",
}

# Coefficients printed by the notebook (fit on the training window; log-RV regression)
COEFS = pd.DataFrame(
    {
        "HAR": [-0.4314, 0.4662, 0.2996, 0.1522, np.nan],
        "HAR + VIX": [-0.1666, 0.4117, 0.2379, -0.0186, 0.3786],
    },
    index=["Intercept", "Yesterday's RV (daily)", "Last 5 days avg (weekly)",
           "Last 22 days avg (monthly)", "Yesterday's India VIX"],
)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_data():
    fc = pd.read_csv(FORECASTS_CSV, index_col=0, parse_dates=True).sort_index()
    mt = pd.read_csv(METRICS_CSV, index_col=0)
    return fc, mt


def to_ann_vol(log_rv):
    """log of daily intraday RV -> annualised volatility in %."""
    return np.exp(log_rv) * math.sqrt(252) * 100


def window_metrics(df, names):
    """log-scale accuracy inside the selected window."""
    y = df["log_rv_actual"].to_numpy()
    sst = float(np.sum((y - y.mean()) ** 2))
    rows = {}
    for n in names:
        e = y - df[MODELS[n]].to_numpy()
        rows[n] = {"log-RMSE": float(np.sqrt(np.mean(e ** 2))),
                   "log-MAE": float(np.mean(np.abs(e))),
                   "R² (log)": 1 - float(np.sum(e ** 2)) / sst}
    return pd.DataFrame(rows).T


def dm_test(e_a, e_b, lags=5):
    """Diebold-Mariano on squared errors, Newey-West variance.
    mean diff > 0  ->  model B has the smaller squared error."""
    d = np.asarray(e_a) ** 2 - np.asarray(e_b) ** 2
    n = len(d)
    mean = d.mean()
    dc = d - mean
    var = float(np.mean(dc ** 2))
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        var += 2 * w * float(np.mean(dc[l:] * dc[:-l]))
    se = math.sqrt(max(var, 1e-18) / n)
    t = mean / se
    p = math.erfc(abs(t) / math.sqrt(2))
    return mean, t, p


def verdict(p, b_better, a_name, b_name):
    if p < 0.05:
        who = b_name if b_better else a_name
        return f"{who} is better (significant at 5%)"
    if p < 0.10:
        who = b_name if b_better else a_name
        return f"{who} looks better, but only weak evidence (p < 0.10)"
    return "No significant difference"


# ----------------------------------------------------------------------------- load
if not FORECASTS_CSV.exists() or not METRICS_CSV.exists():
    st.error(
        "Data files not found. Copy `lstm_direct_test_forecasts.csv` and "
        "`lstm_direct_metrics.csv` from your notebook's `data/processed/` folder "
        "into this app's `data/` folder, then reload."
    )
    st.stop()

fc, metrics = load_data()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
d0, d1 = fc.index.min().date(), fc.index.max().date()
start, end = st.sidebar.slider("Date range", min_value=d0, max_value=d1,
                               value=(d0, d1), format="YYYY-MM-DD")
chosen = st.sidebar.multiselect("Models on charts", list(MODELS), default=list(MODELS))
scale = st.sidebar.radio("Y-axis", ["Annualised volatility (%)", "log RV (model scale)"])
st.sidebar.divider()
st.sidebar.download_button("Download forecasts (CSV)", fc.to_csv().encode(),
                           "volnet_x_test_forecasts.csv", "text/csv")
st.sidebar.caption("Out-of-sample test period only. Nothing here was used for training.")

win = fc.loc[pd.Timestamp(start):pd.Timestamp(end)]
use_log = scale.startswith("log")


def yvals(series):
    return series if use_log else to_ann_vol(series)


# ----------------------------------------------------------------------------- header
st.title("📈 VolNet-X: forecasting NIFTY 50 volatility")
st.markdown(
    "Next-day **realized volatility** forecasts from 1-minute NIFTY 50 data. "
    "A classic econometric model (**HAR**), the same model plus **India VIX**, and a "
    "**hand-written NumPy LSTM** are compared on the same unseen test days "
    f"(**{fc.index.min():%d %b %Y} to {fc.index.max():%d %b %Y}**, {len(fc)} trading days)."
)

main = {m: metrics.loc[METRIC_ROWS[m]] for m in ["HAR", "HAR + VIX", "LSTM (3-seed avg)"]
        if METRIC_ROWS[m] in metrics.index}
if main:
    best = min(main, key=lambda m: main[m]["log-RMSE"])
    cols = st.columns(len(main))
    base = main["HAR"]["log-RMSE"] if "HAR" in main else None
    for col, (name, row) in zip(cols, main.items()):
        delta = None
        if base is not None and name != "HAR":
            delta = f"{(row['log-RMSE'] / base - 1) * 100:+.1f}% vs HAR"
        col.metric(f"{name}: log-RMSE" + ("  🏆" if name == best else ""),
                   f"{row['log-RMSE']:.4f}", delta, delta_color="inverse")
    st.caption("log-RMSE on the full test period. Lower is better. Green arrow = improvement over HAR.")

tab_score, tab_fc, tab_diag, tab_sig, tab_method = st.tabs(
    ["🏁 Scoreboard", "📉 Forecasts", "🔬 Diagnostics", "🧪 Is it significant?", "📘 Method & limits"]
)

# ----------------------------------------------------------------------------- scoreboard
with tab_score:
    st.subheader("Model comparison (full test period)")
    rows = [r for r in METRIC_ROWS.values() if r in metrics.index]
    table = metrics.loc[rows].rename(index={v: k for k, v in METRIC_ROWS.items()})
    try:
        styled = (table.style
                  .format("{:.5f}")
                  .format("{:.3f}", subset=[c for c in table.columns if "R2" in c or "R²" in c])
                  .highlight_min(subset=[c for c in table.columns if "R2" not in c and "R²" not in c],
                                 color="#1b5e20", axis=0)
                  .highlight_max(subset=[c for c in table.columns if "R2" in c or "R²" in c],
                                 color="#1b5e20", axis=0))
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(table.round(5), use_container_width=True)
    st.caption("Green = best in column. RMSE / MAE / QLIKE use RV-scale forecasts (bias-corrected "
               "with a smearing factor); log-RMSE and R² use the log scale.")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **RMSE / MAE**: average size of the miss in raw volatility units. RMSE punishes big misses more.\n"
            "- **QLIKE**: the standard loss for volatility forecasts; it punishes *under*-predicting "
            "volatility more than over-predicting, which is what matters for risk.\n"
            "- **log-RMSE**: error in percentage-like terms, so calm and turbulent days count equally.\n"
            "- **R² (log)**: share of the day-to-day variation in volatility the model explains out of sample. "
            "Daily volatility is very noisy, so even ~0.35 is typical.\n"
            "- **Naive**: just says 'tomorrow = today'. Any useful model must beat it."
        )

    seeds = metrics[metrics.index.str.startswith("LSTM seed")]
    if len(seeds):
        with st.expander("LSTM results vary with the random seed"):
            st.dataframe(seeds.round(5), use_container_width=True)
            spread = seeds["log-RMSE"].max() - seeds["log-RMSE"].min()
            st.caption(f"log-RMSE spread across seeds: {spread:.4f}. A single-seed LSTM result can look "
                       "better or worse than it is, so the dashboard uses the average of all seeds.")

# ----------------------------------------------------------------------------- forecasts
with tab_fc:
    st.subheader("Actual vs forecast volatility")
    if len(win) < 30:
        st.warning("Pick a wider date range (at least 30 trading days).")
    elif not chosen:
        st.info("Select at least one model in the sidebar.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win.index, y=yvals(win["log_rv_actual"]), name="Actual",
                                 mode="lines", line=dict(color=COLORS["Actual"], width=1.4)))
        for m in chosen:
            fig.add_trace(go.Scatter(x=win.index, y=yvals(win[MODELS[m]]), name=m,
                                     mode="lines", line=dict(color=COLORS[m], width=1.2)))
        fig.update_layout(height=480, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="log RV" if use_log else "Annualised volatility (%)",
                          legend=dict(orientation="h", y=1.08))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Accuracy inside the selected window** (log scale)")
        st.dataframe(window_metrics(win, chosen).style.format("{:.4f}")
                     if hasattr(pd.DataFrame, "style") else window_metrics(win, chosen),
                     use_container_width=True)

        st.markdown("**Ten biggest misses of HAR + VIX in this window**")
        miss = win.assign(err=(win["log_rv_actual"] - win["har_vix_linear"]))
        miss = miss.reindex(miss["err"].abs().sort_values(ascending=False).index).head(10)
        out = pd.DataFrame({
            "Actual vol (%)": to_ann_vol(miss["log_rv_actual"]),
            "HAR + VIX (%)": to_ann_vol(miss["har_vix_linear"]),
            "LSTM (%)": to_ann_vol(miss["lstm_direct_avg"]),
            "Underestimated by": (to_ann_vol(miss["log_rv_actual"]) / to_ann_vol(miss["har_vix_linear"]) - 1) * 100,
        }, index=miss.index.strftime("%d %b %Y"))
        out["Underestimated by"] = out["Underestimated by"].map(lambda v: f"{v:+.0f}%")
        st.dataframe(out.round(1), use_container_width=True)
        st.caption("Every model misses sudden spikes: the forecast only sees information up to the previous day.")

# ----------------------------------------------------------------------------- diagnostics
with tab_diag:
    st.subheader("Where does each model do well or badly?")
    if len(win) < 60 or not chosen:
        st.info("Select a window of at least 60 days and at least one model.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Forecast vs actual**")
            pick = st.selectbox("Model", chosen, key="scatter_model")
            xs, ys = win[MODELS[pick]], win["log_rv_actual"]
            lo, hi = float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max()))
            sc = go.Figure()
            sc.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=pick,
                                    marker=dict(size=5, opacity=0.5, color=COLORS[pick])))
            sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect forecast",
                                    line=dict(color="red", dash="dash", width=1)))
            sc.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="forecast (log RV)", yaxis_title="actual (log RV)")
            st.plotly_chart(sc, use_container_width=True)
            st.caption("Points on the dashed line are perfect forecasts. A flatter cloud means the model "
                       "reacts less than reality does.")

        with c2:
            st.markdown("**Rolling 60-day error**")
            roll = go.Figure()
            for m in chosen:
                err2 = (win["log_rv_actual"] - win[MODELS[m]]) ** 2
                roll.add_trace(go.Scatter(x=win.index, y=err2.rolling(60).mean() ** 0.5, name=m,
                                          mode="lines", line=dict(color=COLORS[m], width=1.4)))
            roll.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                               yaxis_title="log-RMSE (60-day rolling)", hovermode="x unified")
            st.plotly_chart(roll, use_container_width=True)
            st.caption("When the lines rise together, the market became harder to predict for every model.")

        st.markdown("**Accuracy by market regime**")
        regime = pd.qcut(win["har"], 3, labels=["Calm", "Normal", "Turbulent"])
        recs = []
        for m in chosen:
            e2 = (win["log_rv_actual"] - win[MODELS[m]]) ** 2
            g = e2.groupby(regime, observed=True).mean() ** 0.5
            recs += [{"Regime": r, "Model": m, "log-RMSE": v} for r, v in g.items()]
        bar = px.bar(pd.DataFrame(recs), x="Regime", y="log-RMSE", color="Model", barmode="group",
                     color_discrete_map=COLORS, category_orders={"Regime": ["Calm", "Normal", "Turbulent"]})
        bar.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, use_container_width=True)
        st.caption("Regimes are the thirds of days ranked by HAR's own forecast, which is known the day before, "
                   "so no hindsight is used to define them.")

# ----------------------------------------------------------------------------- significance
with tab_sig:
    st.subheader("Are the differences real, or just noise?")
    st.markdown(
        "A model can look better on a table by luck. The **Diebold-Mariano test** asks whether one model's "
        "squared errors are *systematically* smaller than another's, using autocorrelation-robust "
        "(Newey-West, 5 lags) standard errors. It uses the full test period."
    )
    err = {m: fc["log_rv_actual"] - fc[MODELS[m]] for m in MODELS}
    pairs = [("HAR", "HAR + VIX"), ("HAR", "LSTM (3-seed avg)"), ("HAR + VIX", "LSTM (3-seed avg)")]
    rows = []
    for a, b in pairs:
        mean, t, p = dm_test(err[a].to_numpy(), err[b].to_numpy())
        rows.append({"Comparison": f"{a}  →  {b}",
                     "Avg. squared-error gain": mean, "t-stat": t, "p-value": p,
                     "Verdict": verdict(p, mean > 0, a, b)})
    st.dataframe(pd.DataFrame(rows).set_index("Comparison").style.format(
        {"Avg. squared-error gain": "{:+.5f}", "t-stat": "{:+.2f}", "p-value": "{:.3f}"}),
        use_container_width=True)
    st.caption("Positive gain = the second model is more accurate. p-values may differ slightly in the "
               "3rd decimal from the notebook (different small-sample handling).")

    st.subheader("What the linear models learned")
    st.dataframe(COEFS.style.format("{:.4f}", na_rep="-"), use_container_width=True)
    st.markdown(
        "Adding **India VIX** takes weight away from the 22-day average. VIX is a forward-looking, "
        "market-implied measure, so it carries information about the slow-moving volatility level that the "
        "monthly average used to supply."
    )

# ----------------------------------------------------------------------------- method
with tab_method:
    st.subheader("How this was built")
    st.markdown(
        """
**Data.** 1-minute NIFTY 50 prices (Jan 2015 to May 2026), cleaned of Muhurat (Diwali evening) sessions and
partial trading days, plus daily India VIX closes.

**Target.** Daily realized volatility: `RV = sqrt(sum of squared 1-minute log returns)` within each session.
Models forecast `log RV`. This measures intraday volatility only (no overnight gap), so it runs lower than a
close-to-close figure.

**Models.**
- **HAR-RV** (Corsi, 2009): linear regression of `log RV` on yesterday's RV, the 5-day average and the 22-day average.
- **HAR + VIX**: the same regression with the previous day's India VIX added.
- **LSTM**: a recurrent network written from scratch in NumPy (forward pass, backpropagation through time,
  Adam, gradient clipping, early stopping), verified against finite-difference gradients. It reads a 22-day
  window of the same four inputs and predicts `log RV` directly. Three random seeds are averaged.

**No look-ahead.** Chronological split (no shuffling). Every input is lagged so only information from before the
forecast day is used. VIX is the previous trading day's close. Scalers are fit on training data only. The
validation set is cut from the end of training, never from the test period.

**Fair comparison.** All models are scored on the same test days with the same metrics.
"""
    )
    st.subheader("Limits you should know about")
    st.warning(
        "- **One test period** (Feb 2024 to May 2026) with no market crash in it, so stress-period behaviour "
        "is not tested here.\n"
        "- **Modest explanatory power.** Daily volatility is noisy; all models miss sharp spikes.\n"
        "- **Small LSTM, small dataset** (~1,900 training windows, 8 hidden units). The result says nothing "
        "about larger networks or richer inputs.\n"
        "- **LSTM seeds disagree** with each other by a margin comparable to the gaps between models.\n"
        "- This is a research showcase, **not investment advice**."
    )
    st.caption("References: Corsi (2009), 'A Simple Approximate Long-Memory Model of Realized Volatility', "
               "Journal of Financial Econometrics.")
