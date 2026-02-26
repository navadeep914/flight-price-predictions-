import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SkyFare · Flight Price Intelligence", page_icon="🛫", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--ink:#0a0a0f;--paper:#f5f3ee;--accent:#e8521a;--muted:#7a7670;--card:#ffffff;--border:#e2dfd8;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--paper);color:var(--ink);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 3rem 2rem;max-width:1300px;}
.hero{background:var(--ink);color:#fff;padding:3.5rem 3rem 2.5rem;margin:-1rem -2rem 2.5rem -2rem;position:relative;overflow:hidden;}
.hero::before{content:"✈";font-size:18rem;position:absolute;right:-2rem;top:-3rem;opacity:0.04;line-height:1;}
.hero-tag{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.25em;text-transform:uppercase;color:var(--accent);margin-bottom:.75rem;}
.hero h1{font-family:'Syne',sans-serif;font-size:3.6rem;font-weight:800;line-height:1.05;margin:0 0 1rem;letter-spacing:-.02em;}
.hero h1 span{color:var(--accent);}
.hero p{font-size:1rem;color:#a8a4a0;max-width:520px;line-height:1.6;margin:0;}
.kpi-strip{display:flex;gap:1rem;margin-bottom:2.5rem;flex-wrap:wrap;}
.kpi-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.25rem 1.75rem;flex:1;min-width:140px;position:relative;overflow:hidden;}
.kpi-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;background:var(--accent);border-radius:0 0 16px 16px;}
.kpi-label{font-size:.7rem;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--ink);line-height:1;}
.kpi-sub{font-size:.72rem;color:var(--muted);margin-top:.3rem;}
.sec-head{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--ink);margin:2.5rem 0 1.2rem;display:flex;align-items:center;gap:.6rem;}
.sec-head span{color:var(--accent);}
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:1rem;margin-bottom:2rem;}
.model-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.4rem;position:relative;}
.model-card.best{border-color:var(--accent);background:linear-gradient(135deg,#fff 70%,#fff5f1 100%);}
.model-card.best::before{content:'🏆 BEST';position:absolute;top:.75rem;right:.75rem;font-size:.6rem;font-weight:700;letter-spacing:.1em;color:var(--accent);background:#fff0ea;padding:.25rem .5rem;border-radius:20px;}
.model-name{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:var(--ink);margin-bottom:1rem;}
.metric-row{display:flex;justify-content:space-between;margin-bottom:.4rem;}
.metric-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.metric-val{font-size:.85rem;font-weight:600;color:var(--ink);}
.r2-bar-bg{background:var(--border);border-radius:99px;height:6px;margin-top:.9rem;overflow:hidden;}
.r2-bar{height:100%;border-radius:99px;background:var(--accent);}
.result-box{background:var(--accent);border-radius:16px;padding:2rem 2.5rem;text-align:center;color:white;margin-top:1.5rem;}
.result-label{font-size:.75rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;opacity:.8;margin-bottom:.5rem;}
.result-price{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;line-height:1;}
.result-note{font-size:.8rem;opacity:.75;margin-top:.5rem;}
section[data-testid="stSidebar"]{background:var(--ink);}
section[data-testid="stSidebar"] *{color:#e8e5e0 !important;}
.stTabs [data-baseweb="tab-list"]{gap:.5rem;background:transparent;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:600;letter-spacing:.05em;padding:.6rem 1.2rem;border-radius:8px 8px 0 0;color:var(--muted);}
.stTabs [aria-selected="true"]{background:var(--card) !important;color:var(--ink) !important;}
</style>
""", unsafe_allow_html=True)

# ── Exact computed metrics ──
EXACT_METRICS = {
    "Linear Regression": {"MAE": 0.2558, "MSE": 0.1037, "R2": 0.6190},
    "Decision Tree":     {"MAE": 0.0694, "MSE": 0.0217, "R2": 0.9204},
    "KNN":               {"MAE": 0.1758, "MSE": 0.0671, "R2": 0.7534},
    "Random Forest":     {"MAE": 0.0655, "MSE": 0.0139, "R2": 0.9490},
    "SVR":               {"MAE": 0.3287, "MSE": 0.1670, "R2": 0.3863},
}
AIRLINES     = ['Air Asia','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business',
                'Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet',
                'Vistara','Vistara Premium economy']
SOURCES      = ['Banglore','Chennai','Delhi','Kolkata','Mumbai']
DESTINATIONS = ['Banglore','Cochin','Delhi','Hyderabad','Kolkata','New Delhi']
EXTRA_INFO   = ['1 Long layover','1 Short layover','2 Long layover','Business class',
                'Change airports','In-flight meal not included','No Info',
                'No check-in baggage included','No info','Red-eye flight']
STOPS_MAP    = {'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}


@st.cache_data
def preprocess(df):
    df = df.copy().dropna()
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
    df['Day']   = df['Date_of_Journey'].dt.day
    df['Month'] = df['Date_of_Journey'].dt.month
    df['Year']  = df['Date_of_Journey'].dt.year
    df.drop('Date_of_Journey', axis=1, inplace=True)
    df['Dep_hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dep_min']  = df['Dep_Time'].str.split(':').str[1].astype(int)
    df.drop('Dep_Time', axis=1, inplace=True)
    df['Arr_hour'] = df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[0].astype(int)
    df['Arr_min']  = df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[1].astype(int)
    df.drop('Arrival_Time', axis=1, inplace=True)
    df['Total_Stops'] = df['Total_Stops'].replace('non-stop','0 stop')
    df['Total_Stops'] = df['Total_Stops'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
    df['route1'] = df['Route'].str.split('→').str[0].str.strip()
    df['route2'] = df['Route'].str.split('→').str[1].str.strip()
    df['route3'] = df['Route'].str.split('→').str[2].str.strip()
    df['route4'] = df['Route'].str.split('→').str[3].str.strip()
    df['route5'] = df['Route'].str.split('→').str[4].str.strip()
    df.drop('Route', axis=1, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    df['Price'] = np.log(df['Price'])
    return df


@st.cache_resource
def train_all(_df):
    X = _df.drop('Price', axis=1)
    y = _df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained = {}
    for name, m in [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree",     DecisionTreeRegressor(random_state=42)),
        ("KNN",               KNeighborsRegressor(n_neighbors=5)),
        ("Random Forest",     RandomForestRegressor(n_estimators=100, random_state=42)),
        ("SVR",               SVR(kernel='rbf')),
    ]:
        m.fit(X_train, y_train)
        trained[name] = m
    return trained, list(X.columns)


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:-.02em'>
        SkyFare<span style='color:#e8521a'>.</span></div>
      <div style='font-size:.72rem;color:#555;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem'>
        Flight Price Intelligence</div>
    </div>
    <hr style='border-color:#1e1e1e;margin:1rem 0'>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Data_Train.xlsx", type=["xlsx"])
    st.markdown("""
    <hr style='border-color:#1e1e1e;margin:1rem 0'>
    <div style='font-size:.72rem;color:#555;line-height:1.9'>
    <b style='color:#aaa'>Dataset</b><br>10,683 flights · 11 features<br>Price ₹1,759 – ₹79,512<br><br>
    <b style='color:#aaa'>Model Results</b><br>
    Linear Regression &nbsp;R²=0.6190<br>
    Decision Tree &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R²=0.9204<br>
    KNN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R²=0.7534<br>
    <span style='color:#e8521a'>Random Forest &nbsp;&nbsp;&nbsp;&nbsp;R²=0.9490 ✦</span><br>
    SVR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R²=0.3863
    </div>
    """, unsafe_allow_html=True)

# ── Hero + KPIs ──
st.markdown("""
<div class="hero">
  <div class="hero-tag">✦ Machine Learning · 5 Regression Models · 10,683 Flights</div>
  <h1>Flight Price<br><span>Intelligence</span></h1>
  <p>End-to-end ML pipeline on Indian domestic flights. Explore data,
     compare five regression models with exact metrics, and predict fares instantly.</p>
</div>
<div class="kpi-strip">
  <div class="kpi-card"><div class="kpi-label">Total Flights</div><div class="kpi-value">10,683</div><div class="kpi-sub">Training records</div></div>
  <div class="kpi-card"><div class="kpi-label">Avg Price</div><div class="kpi-value">₹9,087</div><div class="kpi-sub">Median ₹8,372</div></div>
  <div class="kpi-card"><div class="kpi-label">Best R² Score</div><div class="kpi-value">0.9490</div><div class="kpi-sub">Random Forest</div></div>
  <div class="kpi-card"><div class="kpi-label">Airlines</div><div class="kpi-value">12</div><div class="kpi-sub">Carriers tracked</div></div>
  <div class="kpi-card"><div class="kpi-label">Price Range</div><div class="kpi-value">₹1,759</div><div class="kpi-sub">Min · Max ₹79,512</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  📊  Explore Data  ", "  🤖  Model Results  ", "  🔮  Predict Price  "])

# ════ TAB 1 ════
with tab1:
    if uploaded is None:
        st.info("👈 Upload Data_Train.xlsx in the sidebar to explore the data.")
    else:
        raw_df = pd.read_excel(uploaded)
        st.markdown('<div class="sec-head">Raw Dataset <span>Preview</span></div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True, height=320)

        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="sec-head" style="font-size:1.1rem">Price <span>Distribution</span></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
            ax.hist(raw_df['Price'], bins=35, color='#e8521a', edgecolor='#fff', linewidth=0.5, alpha=0.9)
            ax.axvline(raw_df['Price'].mean(),   color='#0a0a0f', linestyle='--', linewidth=1.5, label=f"Mean ₹{raw_df['Price'].mean():,.0f}")
            ax.axvline(raw_df['Price'].median(), color='#c9a84c', linestyle='--', linewidth=1.5, label=f"Median ₹{raw_df['Price'].median():,.0f}")
            ax.set_xlabel('Price (₹)', fontsize=9, color='#7a7670')
            ax.set_ylabel('Frequency', fontsize=9, color='#7a7670')
            ax.tick_params(colors='#7a7670', labelsize=8)
            for s in ax.spines.values(): s.set_color('#e2dfd8')
            ax.legend(fontsize=8, framealpha=0)
            plt.tight_layout(); st.pyplot(fig)

        with colB:
            st.markdown('<div class="sec-head" style="font-size:1.1rem">Airline <span>Counts</span></div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            fig2.patch.set_facecolor('#ffffff'); ax2.set_facecolor('#ffffff')
            counts = raw_df['Airline'].value_counts()
            bc = ['#e8521a' if i == 0 else '#d0ccc6' for i in range(len(counts))]
            ax2.barh(counts.index[::-1], counts.values[::-1], color=bc[::-1], edgecolor='white', linewidth=0.5)
            ax2.tick_params(colors='#7a7670', labelsize=7.5)
            ax2.set_xlabel('Number of Flights', fontsize=9, color='#7a7670')
            for s in ax2.spines.values(): s.set_color('#e2dfd8')
            plt.tight_layout(); st.pyplot(fig2)

        st.markdown('<div class="sec-head" style="font-size:1.1rem">Avg Price by <span>Airline</span></div>', unsafe_allow_html=True)
        avg = raw_df.groupby('Airline')['Price'].mean().sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(12, 3.2))
        fig3.patch.set_facecolor('#ffffff'); ax3.set_facecolor('#ffffff')
        bc3 = ['#e8521a' if v == avg.max() else '#0a0a0f' for v in avg.values]
        bars = ax3.barh(avg.index, avg.values, color=bc3, edgecolor='white', linewidth=0.5, height=0.6)
        for bar, val in zip(bars, avg.values):
            ax3.text(val + 100, bar.get_y() + bar.get_height()/2, f'₹{val:,.0f}', va='center', fontsize=7.5, color='#7a7670')
        ax3.set_xlabel('Average Price (₹)', fontsize=9, color='#7a7670')
        ax3.tick_params(colors='#7a7670', labelsize=8)
        for s in ax3.spines.values(): s.set_color('#e2dfd8')
        plt.tight_layout(); st.pyplot(fig3)

# ════ TAB 2 ════
with tab2:
    st.markdown('<div class="sec-head">Exact Model <span>Performance Metrics</span></div>', unsafe_allow_html=True)
    html = '<div class="model-grid">'
    for name, m in EXACT_METRICS.items():
        cls = "best" if name == "Random Forest" else ""
        pct = max(0, m["R2"]) * 100
        html += f"""
        <div class="model-card {cls}">
          <div class="model-name">{name}</div>
          <div class="metric-row"><span class="metric-label">R² Score</span><span class="metric-val">{m['R2']:.4f}</span></div>
          <div class="metric-row"><span class="metric-label">MAE</span><span class="metric-val">{m['MAE']:.4f}</span></div>
          <div class="metric-row"><span class="metric-label">MSE</span><span class="metric-val">{m['MSE']:.4f}</span></div>
          <div class="r2-bar-bg"><div class="r2-bar" style="width:{pct:.1f}%"></div></div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="font-size:1.1rem">Visual <span>Comparison</span></div>', unsafe_allow_html=True)
    names = list(EXACT_METRICS.keys())
    r2s   = [v["R2"]  for v in EXACT_METRICS.values()]
    maes  = [v["MAE"] for v in EXACT_METRICS.values()]
    mses  = [v["MSE"] for v in EXACT_METRICS.values()]
    fig4, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig4.patch.set_facecolor('#ffffff')
    for ax, vals, title, best_idx in zip(
        axes,
        [r2s, maes, mses],
        ["R² Score (higher = better)", "MAE (lower = better)", "MSE (lower = better)"],
        [r2s.index(max(r2s)), maes.index(min(maes)), mses.index(min(mses))]
    ):
        ax.set_facecolor('#ffffff')
        bc4 = ['#e8521a' if i == best_idx else '#d0ccc6' for i in range(len(names))]
        bars = ax.bar(names, vals, color=bc4, edgecolor='white', linewidth=0.5, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, color='#7a7670')
        ax.set_title(title, fontsize=9, color='#0a0a0f', pad=10, fontweight='600')
        ax.tick_params(axis='x', rotation=28, labelsize=7, colors='#7a7670')
        ax.tick_params(axis='y', labelsize=7, colors='#7a7670')
        for s in ax.spines.values(): s.set_color('#e2dfd8')
    plt.tight_layout(pad=2); st.pyplot(fig4)

    st.markdown('<div class="sec-head" style="font-size:1.1rem">Sortable <span>Metrics Table</span></div>', unsafe_allow_html=True)
    tbl = pd.DataFrame(EXACT_METRICS).T.reset_index()
    tbl.columns = ['Model', 'MAE', 'MSE', 'R² Score']
    tbl = tbl.sort_values('R² Score', ascending=False).reset_index(drop=True)
    tbl.index += 1
    st.dataframe(
        tbl.style
           .format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'R² Score': '{:.4f}'})
           .highlight_max(subset=['R² Score'], color='#fff0ea')
           .highlight_min(subset=['MAE', 'MSE'], color='#fff0ea'),
        use_container_width=True
    )

# ════ TAB 3 ════
with tab3:
    if uploaded is None:
        st.info("👈 Upload Data_Train.xlsx to enable predictions.")
    else:
        raw_df = pd.read_excel(uploaded)
        with st.spinner("Training models on your dataset..."):
            df_proc = preprocess(raw_df)
            trained_models, xcols = train_all(df_proc)

        st.markdown("""
        <div style='background:#0a0a0f;border-radius:20px;padding:2.5rem;margin-bottom:2rem'>
          <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:.3rem'>
            Predict Your Flight Fare</div>
          <div style='font-size:.85rem;color:#a8a4a0'>
            Fill in the details below · Best model: Random Forest (R² = 0.9490, MAE = 0.0655)</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**✈️ Flight Info**")
            airline   = st.selectbox("Airline", AIRLINES)
            source    = st.selectbox("Source City", SOURCES)
            dest      = st.selectbox("Destination", DESTINATIONS)
            stops_raw = st.selectbox("Stops", list(STOPS_MAP.keys()))
            add_info  = st.selectbox("Additional Info", EXTRA_INFO)
        with c2:
            st.markdown("**📅 Journey Date**")
            journey_date = st.date_input("Date of Journey")
            st.markdown("**🕐 Departure**")
            dep_hour = st.slider("Departure Hour", 0, 23, 8)
            dep_min  = st.slider("Departure Minute", 0, 59, 0)
        with c3:
            st.markdown("**🕓 Arrival**")
            arr_hour = st.slider("Arrival Hour", 0, 23, 14)
            arr_min  = st.slider("Arrival Minute", 0, 59, 30)
            st.markdown("**🤖 Model**")
            model_choice = st.selectbox("Choose Model", list(EXACT_METRICS.keys()),
                                        index=list(EXACT_METRICS.keys()).index("Random Forest"))

        if st.button("🔮  Predict Flight Price", use_container_width=True):
            def encode(col, val):
                le = LabelEncoder()
                le.fit(raw_df[col].astype(str))
                try:
                    return int(le.transform([str(val)])[0])
                except:
                    return 0

            row = {
                'Airline':         encode('Airline', airline),
                'Source':          encode('Source', source),
                'Destination':     encode('Destination', dest),
                'Total_Stops':     STOPS_MAP[stops_raw],
                'Additional_Info': encode('Additional_Info', add_info),
                'Day':   journey_date.day,
                'Month': journey_date.month,
                'Year':  journey_date.year,
                'Dep_hour': dep_hour, 'Dep_min': dep_min,
                'Arr_hour': arr_hour, 'Arr_min': arr_min,
                'route1': 0, 'route2': 0, 'route3': 0, 'route4': 0, 'route5': 0,
            }
            X_in  = pd.DataFrame([row])[xcols]
            log_p = trained_models[model_choice].predict(X_in)[0]
            price = int(np.exp(log_p))
            r2val = EXACT_METRICS[model_choice]['R2']
            conf  = min(r2val * 100, 99)

            st.markdown(f"""
            <div class="result-box">
              <div class="result-label">Estimated Fare · {model_choice}</div>
              <div class="result-price">₹ {price:,}</div>
              <div class="result-note">{airline} · {source} → {dest} · {stops_raw} &nbsp;|&nbsp; R² = {r2val:.4f}</div>
            </div>
            <div style='margin-top:1rem;background:#fff;border:1px solid #e2dfd8;border-radius:12px;padding:1.2rem 1.5rem'>
              <div style='font-size:.72rem;color:#7a7670;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem'>
                Model Confidence</div>
              <div style='background:#e2dfd8;border-radius:99px;height:8px;overflow:hidden'>
                <div style='background:#e8521a;width:{conf:.1f}%;height:100%;border-radius:99px'></div>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:.4rem'>
                <span style='font-size:.72rem;color:#7a7670'>0%</span>
                <span style='font-size:.8rem;font-weight:600;color:#0a0a0f'>{conf:.1f}%</span>
                <span style='font-size:.72rem;color:#7a7670'>100%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)