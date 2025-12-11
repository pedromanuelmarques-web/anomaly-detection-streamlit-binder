# app.py
# Streamlit POC: Anomaly Detection (Training + Detection + Dashboard)
# Author: M365 Copilot
# Description: Upload a training dataset to fit an anomaly detector, then upload an
# evaluation dataset to detect anomalies using ML and statistical rules
# (moving average and volume variance). Visualize results and download outputs.

import io
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import streamlit as st

@dataclass
class TrainedModel:
    model: IsolationForest
    scaler: StandardScaler
    feature_cols: List[str]
    window: int
    ewma_alpha: float
    value_col: str
    ts_col: str
    volume_col: Optional[str]

def ensure_datetime(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col)
    out = out.reset_index(drop=True)
    return out

def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b.replace(0, np.nan).fillna(eps))

def engineer_features(df: pd.DataFrame, ts_col: str, value_col: str, window: int = 30, ewma_alpha: float = 0.2, volume_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = ensure_datetime(df, ts_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if volume_col is not None and volume_col in df.columns:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")
    roll_mean = df[value_col].rolling(window=window, min_periods=max(3, window // 3)).mean()
    roll_std = df[value_col].rolling(window=window, min_periods=max(3, window // 3)).std()
    deviation = df[value_col] - roll_mean
    zscore = _safe_div(deviation, roll_std)
    ewma = df[value_col].ewm(alpha=ewma_alpha, adjust=False).mean()
    diff1 = df[value_col].diff()
    pct_change = df[value_col].pct_change()
    if volume_col and volume_col in df.columns:
        vol_roll_var = df[volume_col].rolling(window=window, min_periods=max(3, window // 3)).var()
    else:
        vol_roll_var = pd.Series(index=df.index, dtype=float)
    feat_df = pd.DataFrame({ts_col: df[ts_col], value_col: df[value_col], "roll_mean": roll_mean, "roll_std": roll_std, "deviation": deviation, "zscore": zscore, "ewma": ewma, "diff1": diff1, "pct_change": pct_change, "vol_roll_var": vol_roll_var})
    feature_cols = [value_col, "roll_mean", "roll_std", "deviation", "zscore", "ewma", "diff1", "pct_change", "vol_roll_var"]
    return feat_df, feature_cols

def fit_model(train_df: pd.DataFrame, ts_col: str, value_col: str, window: int, ewma_alpha: float, volume_col: Optional[str], contamination: float = 0.02, random_state: int = 42) -> TrainedModel:
    feat_df, feature_cols = engineer_features(train_df, ts_col=ts_col, value_col=value_col, window=window, ewma_alpha=ewma_alpha, volume_col=volume_col)
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=300, contamination=contamination, bootstrap=True, random_state=random_state, n_jobs=-1)
    model.fit(Xs)
    return TrainedModel(model=model, scaler=scaler, feature_cols=feature_cols, window=window, ewma_alpha=ewma_alpha, value_col=value_col, ts_col=ts_col, volume_col=volume_col)

def detect_anomalies(model_bundle: TrainedModel, eval_df: pd.DataFrame, ma_sigma_threshold: float = 3.0, vol_sigma_threshold: float = 3.0) -> pd.DataFrame:
    feat_df, feature_cols = engineer_features(eval_df, ts_col=model_bundle.ts_col, value_col=model_bundle.value_col, window=model_bundle.window, ewma_alpha=model_bundle.ewma_alpha, volume_col=model_bundle.volume_col)
    X_eval = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X_eval.isna().any(axis=1)
    X_valid = X_eval[valid_mask]
    scores = np.full(len(feat_df), np.nan)
    preds = np.full(len(feat_df), np.nan)
    if X_valid.shape[0] > 0:
        Xs = model_bundle.scaler.transform(X_valid)
        s_valid = model_bundle.model.score_samples(Xs)
        p_valid = model_bundle.model.predict(Xs)
        scores[valid_mask.values] = s_valid
        preds[valid_mask.values] = p_valid
    out = feat_df.copy()
    out["ml_score"] = scores
    out["ml_pred"] = preds
    out["ml_is_anomaly"] = out["ml_pred"].apply(lambda v: 1 if v == -1 else 0)
    out["ma_is_anomaly"] = (out["zscore"].abs() > ma_sigma_threshold).astype(int)
    if "vol_roll_var" in out.columns and not out["vol_roll_var"].isna().all():
        mu = out["vol_roll_var"].mean()
        sd = out["vol_roll_var"].std()
        var_z = (out["vol_roll_var"] - mu) / (sd if sd and sd > 1e-12 else 1.0)
        out["vol_var_z"] = var_z
        out["vol_is_anomaly"] = (var_z.abs() > vol_sigma_threshold).astype(int)
    else:
        out["vol_var_z"] = np.nan
        out["vol_is_anomaly"] = 0
    out["any_anomaly"] = ((out["ml_is_anomaly"] == 1) | (out["ma_is_anomaly"] == 1) | (out["vol_is_anomaly"] == 1)).astype(int)
    return out

def plot_results(df: pd.DataFrame, ts_col: str, value_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[ts_col], y=df[value_col], mode="lines", name="Value", line=dict(color="#1f77b4")))
    if "roll_mean" in df.columns:
        fig.add_trace(go.Scatter(x=df[ts_col], y=df["roll_mean"], mode="lines", name="Moving Avg", line=dict(color="#ff7f0e", dash="dash")))
    ml_ano = df[df["ml_is_anomaly"] == 1]
    if not ml_ano.empty:
        fig.add_trace(go.Scatter(x=ml_ano[ts_col], y=ml_ano[value_col], mode="markers", name="ML Anomaly", marker=dict(color="#d62728", size=9, symbol="x")))
    ma_ano = df[df["ma_is_anomaly"] == 1]
    if not ma_ano.empty:
        fig.add_trace(go.Scatter(x=ma_ano[ts_col], y=ma_ano[value_col], mode="markers", name="MA Anomaly", marker=dict(color="#9467bd", size=8, symbol="circle-open")))
    if "vol_is_anomaly" in df.columns:
        vol_ano = df[df["vol_is_anomaly"] == 1]
        if not vol_ano.empty:
            fig.add_trace(go.Scatter(x=vol_ano[ts_col], y=vol_ano[value_col], mode="markers", name="Volume Var Anomaly", marker=dict(color="#2ca02c", size=8, symbol="triangle-up")))
    fig.update_layout(title="Anomaly Detection", xaxis_title=ts_col, yaxis_title=value_col, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=520)
    return fig

def summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    ml_cnt = int(df["ml_is_anomaly"].sum()) if "ml_is_anomaly" in df.columns else 0
    ma_cnt = int(df["ma_is_anomaly"].sum()) if "ma_is_anomaly" in df.columns else 0
    vol_cnt = int(df["vol_is_anomaly"].sum()) if "vol_is_anomaly" in df.columns else 0
    any_cnt = int(df["any_anomaly"].sum()) if "any_anomaly" in df.columns else 0
    metrics = pd.DataFrame({"metric": ["Total Points", "ML Anomalies", "MA Anomalies", "Volume Var Anomalies", "Any Anomaly"], "value": [total, ml_cnt, ma_cnt, vol_cnt, any_cnt], "percent": [100.0, (ml_cnt / total * 100) if total else 0.0, (ma_cnt / total * 100) if total else 0.0, (vol_cnt / total * 100) if total else 0.0, (any_cnt / total * 100) if total else 0.0]})
    return metrics

def make_download(df: pd.DataFrame, filename: str) -> Tuple[bytes, str]:
    buff = io.BytesIO()
    df.to_csv(buff, index=False)
    buff.seek(0)
    return buff.read(), filename

def serialize_model(bundle: TrainedModel) -> bytes:
    return pickle.dumps(bundle)

def deserialize_model(blob: bytes) -> TrainedModel:
    return pickle.loads(blob)

def main():
    st.set_page_config(page_title="Anomaly Detection POC", layout="wide")
    st.title("Anomaly Detection POC – Training & Detection")
    st.caption("Upload datasets, configure features, train a model, and detect anomalies.")
    with st.sidebar:
        st.header("1) Upload Datasets")
        train_file = st.file_uploader("Training dataset (.csv)", type=["csv"], key="train")
        eval_file = st.file_uploader("Evaluation dataset (.csv)", type=["csv"], key="eval")
        st.header("2) Configure")
        contamination = st.slider("Expected anomaly rate (contamination)", 0.0, 0.2, 0.02, 0.01)
        window = st.slider("Rolling window (points)", 5, 200, 30, 1)
        ewma_alpha = st.slider("EWMA alpha", 0.01, 0.8, 0.2, 0.01)
        ma_sigma = st.slider("MA z-score threshold", 1.0, 5.0, 3.0, 0.1)
        vol_sigma = st.slider("Volume variance z-score threshold", 1.0, 5.0, 3.0, 0.1)
    def read_csv_safe(file) -> Optional[pd.DataFrame]:
        if file is None:
            return None
        try:
            return pd.read_csv(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, sep=';')
    train_df_raw = read_csv_safe(train_file)
    eval_df_raw = read_csv_safe(eval_file)
    use_synth = False
    if train_df_raw is None or eval_df_raw is None:
        with st.expander("No data? Generate synthetic demo data"):
            use_synth = st.checkbox("Use synthetic data", value=False)
            if use_synth:
                np.random.seed(42)
                n = 500
                dates = pd.date_range("2024-01-01", periods=n, freq="D")
                base = np.sin(np.linspace(0, 12 * np.pi, n)) * 10 + 100
                noise = np.random.normal(0, 1.5, n)
                value = base + noise
                volume = np.random.poisson(lam=200, size=n).astype(float)
                idx_spikes = np.random.choice(np.arange(50, n - 50), size=8, replace=False)
                value[idx_spikes] += np.random.choice([20, -25, 30, -30], size=8)
                volume[idx_spikes] *= np.random.choice([0.3, 0.5, 2.5, 3.0], size=8)
                synth = pd.DataFrame({"timestamp": dates, "value": value, "volume": volume})
                train_df_raw = synth.iloc[:300].copy()
                eval_df_raw = synth.iloc[300:].copy()
                st.success("Synthetic training/evaluation datasets generated.")
    if train_df_raw is not None:
        st.subheader("Training Dataset Preview")
        st.dataframe(train_df_raw.head(10), use_container_width=True)
    if eval_df_raw is not None:
        st.subheader("Evaluation Dataset Preview")
        st.dataframe(eval_df_raw.head(10), use_container_width=True)
    if train_df_raw is not None:
        st.subheader("Configuration: Columns")
        cols = list(train_df_raw.columns)
        ts_default_index = (cols.index('timestamp') if 'timestamp' in cols else 0)
        ts_col = st.selectbox("Timestamp column", options=cols, index=ts_default_index)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df_raw[c])]
        if not numeric_cols:
            numeric_cols = cols
        value_default_index = (numeric_cols.index('value') if 'value' in numeric_cols else 0)
        value_col = st.selectbox("Value/metric column", options=numeric_cols, index=value_default_index)
        volume_default_index = ((numeric_cols.index('volume') + 1) if 'volume' in numeric_cols else 0)
        volume_col = st.selectbox("Volume column (optional)", options=["<none>"] + numeric_cols, index=volume_default_index)
        vol_col_final = None if volume_col == "<none>" else volume_col
        c1, c2 = st.columns(2)
        with c1:
            train_btn = st.button("Train Model", type="primary")
        with c2:
            reset_btn = st.button("Reset Model")
        if reset_btn:
            for k in ["model_bundle", "results_df"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.toast("Model state reset.")
        if train_btn:
            with st.spinner("Training model..."):
                try:
                    bundle = fit_model(train_df_raw, ts_col=ts_col, value_col=value_col, window=window, ewma_alpha=ewma_alpha, volume_col=vol_col_final, contamination=contamination)
                    st.session_state["model_bundle"] = bundle
                    st.success("Model trained successfully.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    if eval_df_raw is not None and "model_bundle" in st.session_state:
        st.subheader("Detect Anomalies on Evaluation Dataset")
        detect_btn = st.button("Run Detection", type="primary")
        if detect_btn:
            with st.spinner("Running detection..."):
                try:
                    res = detect_anomalies(model_bundle=st.session_state["model_bundle"], eval_df=eval_df_raw, ma_sigma_threshold=ma_sigma, vol_sigma_threshold=vol_sigma)
                    st.session_state["results_df"] = res
                    st.success("Detection complete.")
                except Exception as e:
                    st.error(f"Detection failed: {e}")
    if "results_df" in st.session_state:
        res = st.session_state["results_df"]
        st.subheader("Interactive Chart")
        fig = plot_results(res, ts_col=st.session_state["model_bundle"].ts_col, value_col=st.session_state["model_bundle"].value_col)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Summary Metrics")
        st.dataframe(summary_metrics(res), use_container_width=True)
        st.subheader("Top Anomalies (by ML score)")
        top_n = st.slider("Show top N anomalies", 5, 100, 20)
        top_df = res.dropna(subset=["ml_score"]).sort_values("ml_score").head(top_n)
        st.dataframe(top_df[[st.session_state["model_bundle"].ts_col, st.session_state["model_bundle"].value_col, "ml_score", "ml_is_anomaly", "ma_is_anomaly", "vol_is_anomaly"]], use_container_width=True)
        csv_blob, csv_name = make_download(res, "anomaly_results.csv")
        st.download_button("Download Results CSV", data=csv_blob, file_name=csv_name, mime="text/csv")
        st.subheader("Model Persistence")
        model_bytes = serialize_model(st.session_state["model_bundle"])
        st.download_button("Download Trained Model (.pkl)", data=model_bytes, file_name="anomaly_model.pkl")
        uploaded_model = st.file_uploader("Upload pre-trained model (.pkl)", type=["pkl"], key="model_upload")
        if uploaded_model is not None:
            try:
                new_bundle = deserialize_model(uploaded_model.read())
                st.session_state["model_bundle"] = new_bundle
                st.toast("Model replaced with uploaded version.")
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")
    st.markdown("---\n**Notes**\n- The ML model is an Isolation Forest trained on engineered features (moving average, volatility, dynamics, and optional volume variance).\n- Statistical anomaly rules used:\n    - **MA anomaly**: |z-score(value vs. rolling mean)| > threshold\n    - **Volume variance anomaly**: |z-score(rolling variance of volume)| > threshold\n- Adjust the rolling window and thresholds in the sidebar for your data's cadence.\n- Ensure your CSV has a timestamp column and at least one numeric value column.")
if __name__ == "__main__":
    main()
