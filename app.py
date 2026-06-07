"""
🏠 California Housing Price Predictor — Streamlit App
Uses sklearn MLPRegressor (Artificial Neural Network) to predict housing prices.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 ANN Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (dark glassmorphism theme) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a0f1e 100%); }

.hero-title {
    font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin-bottom: .4rem;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 1.1rem; color: #94a3b8; margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a2235, rgba(59,130,246,0.05));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem; text-align: center;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin-bottom: .2rem; }
.metric-label { font-size: .8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .1em; }
.metric-icon  { font-size: 1.8rem; margin-bottom: .4rem; }

.section-hdr {
    font-size: 1.3rem; font-weight: 700; color: #f1f5f9;
    margin: 2rem 0 1rem; display: flex; align-items: center; gap: .5rem;
}

.prediction-box {
    background: linear-gradient(135deg, rgba(16,185,129,.1), rgba(6,182,212,.1));
    border: 2px solid rgba(16,185,129,.4);
    border-radius: 20px; padding: 2rem; text-align: center; margin: 1rem 0;
}
.prediction-value {
    font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.prediction-lbl { font-size: .95rem; color: #94a3b8; margin-top: .5rem; }

.info-box {
    background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.25);
    border-radius: 12px; padding: .9rem 1.4rem; margin: .8rem 0; color: #94a3b8;
}

.badge {
    display: inline-block;
    background: rgba(139,92,246,.15); border: 1px solid rgba(139,92,246,.3);
    border-radius: 8px; padding: .25rem .75rem; font-size: .82rem; color: #c4b5fd; margin: .15rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #94a3b8 !important; }

/* Tabs */
div[data-baseweb="tab-list"] {
    background: #1a2235 !important; border-radius: 12px !important;
    padding: .25rem !important; gap: .25rem !important;
}
button[data-baseweb="tab"] { border-radius: 8px !important; color: #94a3b8 !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; box-shadow: 0 4px 20px rgba(59,130,246,.3);
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 12px; }

hr { border-color: rgba(255,255,255,.06) !important; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor='rgba(26,34,53,0.85)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9')),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
)

def plotly_dark(**extra):
    d = dict(PLOTLY_DARK)
    d.update(extra)
    return d


# ─── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = fetch_california_housing()
    df  = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['MedHouseVal'] = raw.target
    return df, raw


# ─── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(hidden1=64, hidden2=32, max_iter=200, lr=0.001):
    raw     = fetch_california_housing()
    X, y    = raw.data, raw.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc      = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden1, hidden2),
        activation='relu',
        solver='adam',
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
    )
    mlp.fit(X_tr_s, y_tr)

    y_pred = mlp.predict(X_te_s)
    mse  = mean_squared_error(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, y_pred)

    return mlp, sc, y_te, y_pred, mse, mae, rmse, r2


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:3rem'>🏠</div>
        <div style='font-size:1.15rem;font-weight:700;
             background:linear-gradient(135deg,#3b82f6,#8b5cf6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text'>ANN Predictor</div>
        <div style='font-size:.78rem;color:#64748b;margin-top:.25rem'>
            California Housing Prices
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Configuration")
    hidden1  = st.selectbox("Hidden Layer 1 Neurons", [32, 64, 128, 256], index=1)
    hidden2  = st.selectbox("Hidden Layer 2 Neurons", [16, 32, 64, 128], index=1)
    max_iter = st.slider("Max Iterations (Epochs)", 50, 500, 200, step=50)
    lr       = st.select_slider("Learning Rate", [0.0001, 0.001, 0.005, 0.01], value=0.001)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.8rem;color:#475569;line-height:1.9'>
        <b>📦 Dataset:</b> California Housing<br>
        <b>🏗️ Model:</b> Multilayer Perceptron (ANN)<br>
        <b>🔧 Solver:</b> Adam optimiser<br>
        <b>📊 Records:</b> 20,640<br>
        <b>📐 Features:</b> 8
    </div>""", unsafe_allow_html=True)


# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>🏠 California Housing<br>Price Predictor</div>
<div class='hero-sub'>Powered by an Artificial Neural Network (MLP) · sklearn MLPRegressor</div>
""", unsafe_allow_html=True)

st.markdown("""
<span class='badge'>🧠 Deep Learning (ANN)</span>
<span class='badge'>📊 20,640 Samples</span>
<span class='badge'>8 Features</span>
<span class='badge'>⚡ Live Prediction</span>
<span class='badge'>🗺️ Interactive Map</span>
<span class='badge'>📈 Training Curves</span>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── LOAD & TRAIN ──────────────────────────────────────────────────────────────
df, raw_data = load_data()

with st.spinner("⚡ Training ANN model..."):
    model, scaler, y_test, y_pred, mse, mae, rmse, r2 = train_model(hidden1, hidden2, max_iter, lr)


# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Predict", "📉  Training", "📊  Dataset", "🗺️  Map", "🏗️  Architecture"
])


# ═══════════════════════════════════════════════════════════════ TAB 1 — PREDICT
with tab1:
    st.markdown("<div class='section-hdr'>📊 Model Performance on Test Set</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, lbl in [
        (c1, "📉", f"{mse:.4f}", "Mean Squared Error"),
        (c2, "📏", f"{mae:.4f}", "Mean Absolute Error"),
        (c3, "📐", f"{rmse:.4f}", "Root MSE"),
        (c4, "🎯", f"{r2:.4f}",  "R² Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr' style='margin-top:2.5rem'>🔢 Enter House Features</div>", unsafe_allow_html=True)

    FEATURE_META = {
        'MedInc':     ('💰 Median Income',       'Median income in block group (tens of thousands $)'),
        'HouseAge':   ('🏚️ House Age (years)',    'Median house age in block group'),
        'AveRooms':   ('🛏️ Avg Rooms/House',      'Average # of rooms per household'),
        'AveBedrms':  ('🛌 Avg Bedrooms/House',   'Average # of bedrooms per household'),
        'Population': ('👥 Population',           'Block group population'),
        'AveOccup':   ('🏘️ Avg Occupancy',        'Average house occupancy'),
        'Latitude':   ('🌐 Latitude',             'Block group latitude (degrees N)'),
        'Longitude':  ('🌐 Longitude',            'Block group longitude (degrees W)'),
    }

    col_a, col_b = st.columns(2)
    user_vals = {}
    feats = list(FEATURE_META.keys())

    for i, feat in enumerate(feats):
        label, tip = FEATURE_META[feat]
        col = col_a if i < 4 else col_b
        step = 1.0 if feat in ('HouseAge', 'Population') else 0.01
        with col:
            user_vals[feat] = st.number_input(
                label,
                min_value=float(df[feat].min()),
                max_value=float(df[feat].max()),
                value=float(round(df[feat].median(), 2)),
                step=step,
                help=tip,
                key=f"inp_{feat}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict House Price", use_container_width=True, key="pred_btn"):
        X_in  = np.array([[user_vals[f] for f in feats]])
        X_sc  = scaler.transform(X_in)
        pred  = model.predict(X_sc)[0]
        price = pred * 100_000

        st.markdown(f"""
        <div class='prediction-box'>
            <div style='font-size:1rem;color:#94a3b8;margin-bottom:.5rem'>
                🏡 Predicted Median House Value
            </div>
            <div class='prediction-value'>${price:,.0f}</div>
            <div class='prediction-lbl'>Model output: {pred:.4f} (×$100k)</div>
        </div>""", unsafe_allow_html=True)

        med_price = df['MedHouseVal'].median() * 100_000
        diff_pct  = (price / med_price - 1) * 100
        direction = "above" if price > med_price else "below"
        st.markdown(f"""
        <div class='info-box'>
            📊 <b>Context:</b> Dataset median house value is <b>${med_price:,.0f}</b>.
            Your prediction is <b>{diff_pct:+.1f}%</b> {direction} the median.
        </div>""", unsafe_allow_html=True)

        # Radar chart
        normed = [(user_vals[f] - df[f].min()) / (df[f].max() - df[f].min()) for f in feats]
        labels  = [FEATURE_META[f][0] for f in feats]
        fig_r = go.Figure(go.Scatterpolar(
            r=normed, theta=labels, fill='toself',
            fillcolor='rgba(59,130,246,.15)',
            line=dict(color='#3b82f6', width=2),
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            **plotly_dark(height=360,
                          title=dict(text='Feature Profile (normalised)', font=dict(color='#f1f5f9', size=14)),
                          showlegend=False)
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════ TAB 2 — TRAINING
with tab2:
    st.markdown("<div class='section-hdr'>📉 Training Convergence</div>", unsafe_allow_html=True)

    loss_curve = model.loss_curve_
    val_curve  = model.validation_scores_ if hasattr(model, 'validation_scores_') and model.validation_scores_ else None
    iters = list(range(1, len(loss_curve) + 1))

    col1, col2 = st.columns(2)

    with col1:
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(
            x=iters, y=loss_curve, name='Training Loss',
            line=dict(color='#3b82f6', width=2.5)
        ))
        fig_lc.update_layout(
            title='📉 MSE Loss over Iterations',
            xaxis_title='Iteration', yaxis_title='MSE Loss',
            **plotly_dark(height=360)
        )
        st.plotly_chart(fig_lc, use_container_width=True)

    with col2:
        # Predicted vs actual scatter (mini version)
        fig_sv = go.Figure()
        fig_sv.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(
                color=np.abs(y_test - y_pred),
                colorscale='Viridis',
                size=4, opacity=0.65,
                colorbar=dict(title='|Error|')
            ), name='Predictions'
        ))
        lim = float(max(y_test.max(), y_pred.max()))
        fig_sv.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode='lines',
            line=dict(color='#ef4444', dash='dash', width=2),
            name='Perfect'
        ))
        fig_sv.update_layout(
            title='🔵 Predicted vs Actual',
            xaxis_title='Actual (×$100k)', yaxis_title='Predicted (×$100k)',
            **plotly_dark(height=360)
        )
        st.plotly_chart(fig_sv, use_container_width=True)

    # Residuals histogram
    st.markdown("<div class='section-hdr'>📊 Residuals Distribution</div>", unsafe_allow_html=True)
    residuals = y_test - y_pred
    fig_res = go.Figure(go.Histogram(
        x=residuals, nbinsx=70,
        marker=dict(color='rgba(139,92,246,.7)', line=dict(color='rgba(139,92,246,1)', width=1))
    ))
    fig_res.add_vline(x=0, line_dash='dash', line_color='#ef4444', line_width=2)
    fig_res.add_vline(x=residuals.mean(), line_dash='dot', line_color='#f59e0b', line_width=1.5,
                      annotation_text=f"μ={residuals.mean():.3f}", annotation_font_color='#f59e0b')
    fig_res.update_layout(
        title='Residuals (Actual − Predicted)', xaxis_title='Residual', yaxis_title='Count',
        **plotly_dark(height=360)
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Error metrics bar
    st.markdown("<div class='section-hdr'>📏 Error by Decile</div>", unsafe_allow_html=True)
    df_err = pd.DataFrame({'actual': y_test, 'pred': y_pred, 'error': np.abs(y_test - y_pred)})
    df_err['decile'] = pd.qcut(df_err['actual'], 10, labels=[f"D{i}" for i in range(1, 11)])
    dec_err = df_err.groupby('decile', observed=True)['error'].mean().reset_index()
    fig_dec = px.bar(
        dec_err, x='decile', y='error',
        color='error', color_continuous_scale='Plasma',
        labels={'error': 'Mean |Error| (×$100k)', 'decile': 'Price Decile'},
        title='Mean Absolute Error by Price Decile'
    )
    fig_dec.update_layout(**plotly_dark(height=360, showlegend=False))
    st.plotly_chart(fig_dec, use_container_width=True)


# ══════════════════════════════════════════════════════════════ TAB 3 — DATASET
with tab3:
    st.markdown("<div class='section-hdr'>📊 Dataset Explorer</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, val, lbl in [
        (c1, "📁", "20,640", "Total Records"),
        (c2, "📐", "8",      "Input Features"),
        (c3, "🎯", "1",      "Target Variable"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📄 Raw Data Sample (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True, height=280)

    with st.expander("📈 Descriptive Statistics"):
        st.dataframe(df.describe().T, use_container_width=True)

    # Feature distribution
    st.markdown("<div class='section-hdr'>📊 Feature Distribution</div>", unsafe_allow_html=True)
    feat_sel = st.selectbox("Select Feature", df.columns.tolist(), index=0, key="fd_sel")
    fig_fd = px.histogram(df, x=feat_sel, nbins=60,
                          color_discrete_sequence=['#3b82f6'], opacity=.8, marginal='box')
    fig_fd.update_layout(**plotly_dark(height=380))
    st.plotly_chart(fig_fd, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-hdr'>🔥 Correlation Matrix</div>", unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2), texttemplate='%{text}',
        textfont=dict(size=10),
    ))
    fig_c.update_layout(**plotly_dark(height=480))
    st.plotly_chart(fig_c, use_container_width=True)

    # Feature vs target
    st.markdown("<div class='section-hdr'>🔵 Feature vs. Target</div>", unsafe_allow_html=True)
    feat_x = st.selectbox("X-axis", [c for c in df.columns if c != 'MedHouseVal'], index=0, key="fvt_x")
    samp   = df.sample(3000, random_state=42)
    fig_fvt = px.scatter(
        samp, x=feat_x, y='MedHouseVal',
        color='MedHouseVal', color_continuous_scale='Viridis',
        opacity=.6,
        labels={'MedHouseVal': 'Med House Val (×$100k)'},
        title=f'{feat_x} vs Median House Value (3 k sample)'
    )
    fig_fvt.update_layout(**plotly_dark(height=440))
    st.plotly_chart(fig_fvt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════ TAB 4 — MAP
with tab4:
    st.markdown("<div class='section-hdr'>🗺️ California Housing Map</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        🌍 Housing prices visualised geographically across California.
        Point colour & size = median house value. (5 k sample)
    </div>""", unsafe_allow_html=True)

    smap = df.sample(5000, random_state=42)

    fig_map = px.scatter_mapbox(
        smap, lat='Latitude', lon='Longitude',
        color='MedHouseVal', size='MedHouseVal',
        color_continuous_scale='Plasma', size_max=10,
        zoom=5, center=dict(lat=37.5, lon=-119.5),
        mapbox_style='carto-darkmatter', opacity=.75,
        hover_data={'MedHouseVal': ':.2f', 'MedInc': ':.2f',
                    'HouseAge': True, 'Population': True},
        labels={'MedHouseVal': 'Med Val (×$100k)', 'MedInc': 'Median Income'},
        title='California Housing Prices'
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(26,34,53,0.85)', font=dict(color='#94a3b8'),
        height=600, margin=dict(r=0, l=0, t=40, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div class='section-hdr'>🔥 Price Density Heatmap</div>", unsafe_allow_html=True)
    fig_dm = px.density_mapbox(
        smap, lat='Latitude', lon='Longitude', z='MedHouseVal',
        radius=15, zoom=5, center=dict(lat=37.5, lon=-119.5),
        mapbox_style='carto-darkmatter', color_continuous_scale='Inferno',
        title='Housing Value Density'
    )
    fig_dm.update_layout(
        paper_bgcolor='rgba(26,34,53,0.85)', font=dict(color='#94a3b8'),
        height=540, margin=dict(r=0, l=0, t=40, b=0)
    )
    st.plotly_chart(fig_dm, use_container_width=True)


# ══════════════════════════════════════════════════════════ TAB 5 — ARCHITECTURE
with tab5:
    st.markdown("<div class='section-hdr'>🏗️ ANN Architecture</div>", unsafe_allow_html=True)

    LAYERS = [
        ("🔵 Input Layer",    f"{len(raw_data.feature_names)} neurons", "8 housing features", "#3b82f6"),
        ("🟣 Hidden Layer 1", f"{hidden1} neurons",                     "ReLU activation",    "#8b5cf6"),
        ("🟤 Hidden Layer 2", f"{hidden2} neurons",                     "ReLU activation",    "#f59e0b"),
        ("🟢 Output Layer",   "1 neuron",                               "Linear (regression)","#10b981"),
    ]
    for lbl, neurons, act, clr in LAYERS:
        st.markdown(f"""
        <div style='background:rgba(0,0,0,.2);border:1px solid {clr}44;
                    border-left:4px solid {clr};border-radius:12px;
                    padding:.9rem 1.4rem;margin:.4rem 0;
                    display:flex;align-items:center;justify-content:space-between'>
            <span style='font-weight:600;color:#f1f5f9;font-size:1rem'>{lbl}</span>
            <span style='font-weight:700;color:{clr};font-size:1.1rem'>{neurons}</span>
            <span style='color:#94a3b8;font-size:.9rem'>{act}</span>
        </div>""", unsafe_allow_html=True)

    # Config row
    st.markdown("<div class='section-hdr'>⚙️ Training Configuration</div>", unsafe_allow_html=True)
    cfg_cols = st.columns(4)
    for col, lbl, val in [
        (cfg_cols[0], "🔧 Solver",        "Adam"),
        (cfg_cols[1], "📉 Loss",          "MSE"),
        (cfg_cols[2], "🔄 Max Iterations", str(max_iter)),
        (cfg_cols[3], "📚 Learning Rate",  str(lr)),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:1.3rem'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Data splits donut
    st.markdown("<div class='section-hdr'>📊 Data Splits</div>", unsafe_allow_html=True)
    total  = len(df)
    tr_sz  = int(total * .8 * .8)
    va_sz  = int(total * .8 * .2)
    te_sz  = int(total * .2)
    fig_sp = go.Figure(go.Pie(
        labels=['Training', 'Validation', 'Test'],
        values=[tr_sz, va_sz, te_sz],
        hole=.55,
        marker=dict(colors=['#3b82f6', '#8b5cf6', '#ec4899']),
        textinfo='label+percent', textfont=dict(color='white', size=12)
    ))
    fig_sp.add_annotation(
        text=f"<b>{total:,}</b><br>samples",
        x=.5, y=.5, font=dict(size=16, color='#f1f5f9'), showarrow=False
    )
    fig_sp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
        height=340, showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'))
    )
    st.plotly_chart(fig_sp, use_container_width=True)

    # Metrics table
    st.markdown("<div class='section-hdr'>📈 Final Test Metrics</div>", unsafe_allow_html=True)
    met_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE', 'R²'],
        'Value' : [f'{mse:.4f}', f'{mae:.4f}', f'{rmse:.4f}', f'{r2:.4f}'],
        'Meaning': [
            'Penalises large errors heavily (lower is better)',
            f'Average prediction error ≈ ${mae*100_000:,.0f}',
            f'Typical error magnitude ≈ ${rmse*100_000:,.0f}',
            f'Model explains {r2*100:.1f}% of price variance',
        ]
    })
    st.dataframe(met_df, use_container_width=True, hide_index=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#475569;font-size:.82rem;padding:.8rem 0'>
    🏠 California Housing ANN Predictor &nbsp;·&nbsp;
    Built with ❤️ using scikit-learn MLPRegressor & Streamlit &nbsp;·&nbsp;
    Data: <b>sklearn.datasets.fetch_california_housing</b>
</div>""", unsafe_allow_html=True)
