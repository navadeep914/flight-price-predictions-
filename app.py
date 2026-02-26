"""
✈️ Flight Price Prediction Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os
import warnings
from datetime import date, time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Prediction Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');
:root{--bg:#07090f;--surf:#0d1117;--surf2:#131b27;--border:#1c2a3a;
      --cyan:#00d4ff;--orange:#ff6b35;--green:#00e5a0;--gold:#f5c842;
      --purple:#a78bfa;--muted:#4a5a72;--text:#dde4ef;--white:#ffffff;}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;background:var(--bg)!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 1.8rem 4rem!important;max-width:1380px!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px;}

section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
section[data-testid="stSidebar"] .stRadio label{padding:.65rem 1rem!important;border-radius:10px!important;
  font-size:.88rem!important;font-weight:500!important;color:var(--muted)!important;transition:all .2s!important;}
section[data-testid="stSidebar"] .stRadio label:hover{background:var(--surf2)!important;color:var(--text)!important;}

.hero-banner{background:linear-gradient(135deg,#07090f,#0a1628,#07090f);border-bottom:1px solid var(--border);
  padding:2.2rem 2rem 1.8rem;margin:-1rem -1.8rem 2rem -1.8rem;position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 55% 90% at 85% 50%,rgba(0,212,255,.07),transparent 70%);}
.hero-banner::after{content:'✈';position:absolute;right:3rem;top:50%;transform:translateY(-50%);
  font-size:9rem;opacity:.04;line-height:1;}
.hero-inner{position:relative;z-index:1;}
.hero-eyebrow{font-size:.65rem;font-weight:700;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);margin-bottom:.5rem;}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(2rem,4vw,3.4rem);letter-spacing:.06em;
  color:var(--white);line-height:1;margin:0 0 .5rem;}
.hero-title span{color:var(--cyan);}
.hero-desc{font-size:.88rem;color:var(--muted);max-width:560px;line-height:1.6;}

.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:.75rem;margin-bottom:2rem;}
.kpi-card{background:var(--surf);border:1px solid var(--border);border-radius:14px;
  padding:1.1rem 1.2rem;position:relative;overflow:hidden;transition:border-color .2s;}
.kpi-card:hover{border-color:var(--cyan);}
.kpi-accent{position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:14px 0 0 14px;}
.kpi-label{font-size:.62rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:.35rem;}
.kpi-value{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;letter-spacing:.04em;color:var(--white);line-height:1;}
.kpi-sub{font-size:.65rem;color:var(--muted);margin-top:.2rem;}

.sec-title{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:.1em;color:var(--white);
  display:flex;align-items:center;gap:.5rem;margin:0 0 1rem;padding-bottom:.6rem;border-bottom:1px solid var(--border);}
.sec-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}

.feat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.75rem;margin-top:.8rem;}
.feat-card{background:var(--surf2);border:1px solid var(--border);border-radius:12px;
  padding:1.1rem 1.2rem;transition:border-color .2s,transform .2s;}
.feat-card:hover{border-color:var(--cyan);transform:translateY(-2px);}
.feat-icon{font-size:1.5rem;margin-bottom:.5rem;}
.feat-name{font-weight:600;font-size:.9rem;color:var(--white);margin-bottom:.25rem;}
.feat-desc{font-size:.75rem;color:var(--muted);line-height:1.5;}

.predict-form{background:var(--surf2);border:1px solid var(--border);border-radius:16px;padding:1.6rem;}
.form-section-label{font-size:.65rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:var(--cyan);margin-bottom:.75rem;display:block;}

.result-box{background:linear-gradient(135deg,#0a1f35,#0d2840);border:1px solid var(--cyan);
  border-radius:16px;padding:2rem 2.5rem;text-align:center;position:relative;overflow:hidden;}
.result-box::before{content:'₹';position:absolute;font-family:'Bebas Neue',sans-serif;font-size:12rem;
  right:-1rem;top:-1rem;opacity:.03;color:var(--cyan);line-height:1;}
.result-label{font-size:.65rem;font-weight:700;letter-spacing:.25em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;}
.result-price{font-family:'Bebas Neue',sans-serif;font-size:4.5rem;letter-spacing:.04em;color:var(--cyan);line-height:1;}
.result-note{font-size:.78rem;color:var(--muted);margin-top:.4rem;}
.result-conf{margin-top:1rem;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);
  border-radius:10px;padding:.75rem 1rem;font-size:.78rem;color:var(--muted);}

.stButton>button{background:linear-gradient(135deg,#00d4ff,#0099cc)!important;color:#000!important;
  font-family:'Outfit',sans-serif!important;font-weight:700!important;font-size:.85rem!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;border:none!important;
  border-radius:10px!important;padding:.75rem 1.5rem!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.85!important;}

.tools-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.6rem;margin-top:.8rem;}
.tool-pill{background:var(--surf2);border:1px solid var(--border);border-radius:8px;
  padding:.6rem 1rem;font-size:.8rem;font-weight:500;color:var(--text);text-align:center;}
.tool-pill span{font-size:1.1rem;display:block;margin-bottom:.2rem;}

.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid var(--border);gap:.2rem;}
.stTabs [data-baseweb="tab"]{font-family:'Outfit',sans-serif;font-size:.78rem;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;padding:.55rem 1.2rem;
  border-radius:8px 8px 0 0;color:var(--muted)!important;}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;
  border-bottom:2px solid var(--cyan)!important;background:rgba(0,212,255,.05)!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#1c2a3a",
    "axes.labelcolor":  "#4a5a72",
    "xtick.color":      "#4a5a72",
    "ytick.color":      "#4a5a72",
    "text.color":       "#dde4ef",
    "grid.color":       "#1c2a3a",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

CYAN   = "#00d4ff"
ORANGE = "#ff6b35"
GREEN  = "#00e5a0"
GOLD   = "#f5c842"
PURPLE = "#a78bfa"
DIM    = "#1c2a3a"

# ─────────────────────────────────────────────
#  CONSTANTS — match training exactly
# ─────────────────────────────────────────────
# These are the 11 features the RF model was trained on (in order):
FEATURE_COLS = [
    "Airline", "Source", "Destination", "Total_Stops",
    "Journey_Day", "Journey_Month",
    "Dep_hour", "Dep_min",
    "Arr_hour", "Arr_min",
    "Duration_mins",       # ← key fix: Duration_mins, NOT Duration
]

AIRLINES     = ['Air Asia','Air India','GoAir','IndiGo','Jet Airways',
                'Jet Airways Business','Multiple carriers',
                'Multiple carriers Premium economy','SpiceJet','Trujet',
                'Vistara','Vistara Premium economy']
SOURCES      = ['Banglore','Chennai','Delhi','Kolkata','Mumbai']
DESTINATIONS = ['Banglore','Cochin','Delhi','Hyderabad','Kolkata','New Delhi']
STOPS_OPTS   = ['non-stop','1 stop','2 stops','3 stops','4 stops']

DATA_PATHS   = ["Data_Train.xlsx",
                "/mnt/user-data/uploads/Data_Train.xlsx",
                "/mnt/data/Data_Train.xlsx"]
MODEL_PATHS  = ["flight_price_model.pkl",
                "/mnt/user-data/outputs/flight_price_model.pkl",
                "/mnt/data/flight_price_model.pkl"]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def dur_to_mins(d):
    """Convert duration string like '2h 30m' to integer minutes."""
    try:
        h = int(str(d).split("h")[0].strip()) if "h" in str(d) else 0
        m = int(str(d).split("h")[-1].replace("m","").strip()) if "m" in str(d) else 0
        return h * 60 + m
    except:
        return 0


def safe_encode(le, val):
    """Label-encode a value; return 0 if unseen."""
    return int(le.transform([val])[0]) if val in le.classes_ else 0


# ─────────────────────────────────────────────
#  DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return pd.read_excel(p)
    return None


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Load pre-trained model pkl. Falls back to training from scratch."""
    for p in MODEL_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    # Train from scratch if pkl missing
    data_path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if data_path is None:
        return None
    return _train_model(data_path)


def _train_model(data_path):
    """Preprocess dataset and train a Random Forest model."""
    df = pd.read_excel(data_path)
    df.dropna(inplace=True)

    # Date features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"]     = df["Date_of_Journey"].dt.day
    df["Journey_Month"]   = df["Date_of_Journey"].dt.month
    df.drop("Date_of_Journey", axis=1, inplace=True)

    # Departure time
    df["Dep_hour"] = df["Dep_Time"].str.split(":").str[0].astype(int)
    df["Dep_min"]  = df["Dep_Time"].str.split(":").str[1].astype(int)
    df.drop("Dep_Time", axis=1, inplace=True)

    # Arrival time
    df["Arr_hour"] = df["Arrival_Time"].str.split(" ").str[0].str.split(":").str[0].astype(int)
    df["Arr_min"]  = df["Arrival_Time"].str.split(" ").str[0].str.split(":").str[1].astype(int)
    df.drop("Arrival_Time", axis=1, inplace=True)

    # Stops → integer
    df["Total_Stops"] = df["Total_Stops"].replace("non-stop", "0 stop")
    df["Total_Stops"] = df["Total_Stops"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)

    # Duration → minutes  (stored as Duration_mins)
    df["Duration_mins"] = df["Duration"].apply(dur_to_mins)
    df.drop(["Duration", "Route", "Additional_Info"], axis=1, inplace=True)

    # Encode categoricals
    le_a = LabelEncoder(); df["Airline"]     = le_a.fit_transform(df["Airline"])
    le_s = LabelEncoder(); df["Source"]      = le_s.fit_transform(df["Source"])
    le_d = LabelEncoder(); df["Destination"] = le_d.fit_transform(df["Destination"])

    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Ensure column order matches FEATURE_COLS
    X = X[FEATURE_COLS]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X_train, y_train)

    return {
        "model":        mdl,
        "le_airline":   le_a,
        "le_source":    le_s,
        "le_dest":      le_d,
        "feature_cols": FEATURE_COLS,   # always use constant — no drift
    }


def build_input_row(airline, source, dest, stops_str,
                    journey_date, dep_time, arr_time, dur_mins,
                    le_airline, le_source, le_dest):
    """Build a single-row DataFrame matching FEATURE_COLS exactly."""
    stops_num = int(stops_str.replace("non-stop","0").split()[0])

    row = {
        "Airline":       safe_encode(le_airline, airline),
        "Source":        safe_encode(le_source,  source),
        "Destination":   safe_encode(le_dest,    dest),
        "Total_Stops":   stops_num,
        "Journey_Day":   journey_date.day,
        "Journey_Month": journey_date.month,
        "Dep_hour":      dep_time.hour,
        "Dep_min":       dep_time.minute,
        "Arr_hour":      arr_time.hour,
        "Arr_min":       arr_time.minute,
        "Duration_mins": dur_mins,
    }
    return pd.DataFrame([row])[FEATURE_COLS]   # strict column order


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .3rem'>
      <div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;letter-spacing:.06em;color:#fff'>
        SKY<span style='color:#00d4ff'>FARE</span></div>
      <div style='font-size:.62rem;color:#4a5a72;letter-spacing:.2em;text-transform:uppercase'>
        Prediction Dashboard</div>
    </div>
    <div style='height:1px;background:#1c2a3a;margin:.8rem 0 1.2rem'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", ["🏠  Home","📊  Data Overview","🔮  Model Prediction","ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='height:1px;background:#1c2a3a;margin:1.2rem 0'></div>
    <div style='font-size:.65rem;color:#4a5a72;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.6rem'>
      Quick Stats</div>
    <div style='font-size:.78rem;line-height:2;color:#6a7a92'>
      📦 10,683 flight records<br>🛫 12 airlines<br>🏙️ 5 source cities<br>
      🎯 6 destinations<br>💰 ₹1,759 – ₹79,512<br>🤖 Random Forest
    </div>
    <div style='height:1px;background:#1c2a3a;margin:1rem 0'></div>
    <div style='background:#0a1f35;border:1px solid #00d4ff33;border-radius:10px;padding:.9rem;text-align:center'>
      <div style='font-size:.6rem;color:#4a5a72;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.2rem'>
        Best Model</div>
      <div style='font-family:Bebas Neue,sans-serif;font-size:1.8rem;color:#00d4ff;letter-spacing:.04em;line-height:1'>
        R² = 0.9490</div>
      <div style='font-size:.65rem;color:#4a5a72;margin-top:.2rem'>Random Forest · MAE=0.0655</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
PAGE_META = {
    "🏠  Home":             ("HOME",           "Welcome to the Flight Price Intelligence System"),
    "📊  Data Overview":    ("DATA OVERVIEW",  "Explore the training dataset in depth"),
    "🔮  Model Prediction": ("LIVE PREDICTION","Enter flight details and get an instant fare estimate"),
    "ℹ️  About":            ("ABOUT",          "Project overview, model results, tools and author"),
}
eyebrow, subtitle = PAGE_META[page]

st.markdown(f"""
<div class="hero-banner">
  <div class="hero-inner">
    <div class="hero-eyebrow">✦ SkyFare Intelligence · {eyebrow}</div>
    <div class="hero-title">✈️ Flight Price <span>Prediction</span></div>
    <div class="hero-desc">{subtitle}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-accent" style="background:#00d4ff"></div>
        <div class="kpi-label">Total Records</div><div class="kpi-value">10,683</div>
        <div class="kpi-sub">Training flights</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#ff6b35"></div>
        <div class="kpi-label">Avg Fare</div><div class="kpi-value">₹9,087</div>
        <div class="kpi-sub">Median ₹8,372</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#00e5a0"></div>
        <div class="kpi-label">Best R² Score</div><div class="kpi-value">0.9490</div>
        <div class="kpi-sub">Random Forest</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#f5c842"></div>
        <div class="kpi-label">Airlines</div><div class="kpi-value">12</div>
        <div class="kpi-sub">Carriers covered</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#a78bfa"></div>
        <div class="kpi-label">Price Range</div><div class="kpi-value">₹79K</div>
        <div class="kpi-sub">₹1,759 – ₹79,512</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00d4ff"></span>About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:.9rem;line-height:1.9;color:#8a9ab2'>
        <b style='color:#dde4ef'>SkyFare Intelligence</b> is an ML-powered flight price prediction system for
        Indian domestic routes. Using <b style='color:#dde4ef'>10,683 flight records</b> across 12 airlines,
        the model learns pricing patterns and delivers instant estimates.<br><br>
        The deployed engine — <b style='color:#00d4ff'>Random Forest Regression</b> — achieves
        R²&nbsp;=&nbsp;<b style='color:#00d4ff'>0.9490</b>, explaining 94.9&nbsp;% of price variance.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#ff6b35"></span>How It Works</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1","Input",      "Select airline, route, stops, date & times"),
            ("2","Preprocess", "Day/month/hour features extracted; categoricals encoded"),
            ("3","Predict",    "Random Forest outputs a fare estimate"),
            ("4","Display",    "Price shown in ₹ with feature summary"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:1rem;align-items:flex-start;margin-bottom:.9rem'>
              <div style='background:#0a1f35;border:1px solid #00d4ff33;border-radius:8px;
                          width:2rem;height:2rem;display:flex;align-items:center;justify-content:center;
                          font-family:Bebas Neue,sans-serif;font-size:1rem;color:#00d4ff;flex-shrink:0'>{num}</div>
              <div>
                <div style='font-weight:600;font-size:.88rem;color:#dde4ef'>{title}</div>
                <div style='font-size:.76rem;color:#6a7a92;margin-top:.1rem'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00e5a0"></span>Key Features</div>', unsafe_allow_html=True)
        feats = [
            ("📊","Rich Data Exploration","Dataset preview, stats, missing values, charts"),
            ("🤖","ML-Powered Prediction","Random Forest · 100 trees · 8,546 training samples"),
            ("📈","Visual Analytics",     "Price by airline, source, stops, model comparison"),
            ("⚡","Instant Results",      "Real-time ₹ prediction on button click"),
            ("🎯","High Accuracy",        "R²=0.9490 · MAE=0.0655 on test set"),
            ("🧹","Smart Preprocessing", "Auto feature engineering from raw inputs"),
        ]
        st.markdown('<div class="feat-grid">', unsafe_allow_html=True)
        for icon, name, desc in feats:
            st.markdown(f'<div class="feat-card"><div class="feat-icon">{icon}</div>'
                        f'<div class="feat-name">{name}</div>'
                        f'<div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#f5c842"></span>Airline Average Fare</div>', unsafe_allow_html=True)
    airline_avg = {
        "Jet Airways Business":58359,"Jet Airways":11644,"Multi Carriers Prem":11419,
        "Multiple Carriers":10903,"Air India":9612,"Vistara Prem":8962,"Vistara":7796,
        "GoAir":5861,"IndiGo":5674,"Air Asia":5590,"SpiceJet":4338,"Trujet":4140,
    }
    s = dict(sorted(airline_avg.items(), key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    bc = [CYAN if v == max(s.values()) else ORANGE if v > 9000 else DIM for v in s.values()]
    bars = ax.barh(list(s.keys()), list(s.values()), color=bc, height=0.62, edgecolor="#07090f", linewidth=0.4)
    for bar, val in zip(bars, s.values()):
        ax.text(val+300, bar.get_y()+bar.get_height()/2, f"₹{val:,}", va="center", fontsize=7.5, color="#6a7a92")
    ax.set_xlabel("Average Price (₹)", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
#  PAGE: DATA OVERVIEW
# ══════════════════════════════════════════════
elif page == "📊  Data Overview":

    df = load_data()
    if df is None:
        st.error("⚠️ Dataset not found. Place Data_Train.xlsx next to app.py.")
        st.stop()

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-accent" style="background:#00d4ff"></div>
        <div class="kpi-label">Rows</div><div class="kpi-value">{df.shape[0]:,}</div><div class="kpi-sub">Flight records</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#ff6b35"></div>
        <div class="kpi-label">Columns</div><div class="kpi-value">{df.shape[1]}</div><div class="kpi-sub">Raw features</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#00e5a0"></div>
        <div class="kpi-label">Missing Values</div><div class="kpi-value">{int(df.isnull().sum().sum())}</div><div class="kpi-sub">Total nulls</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#f5c842"></div>
        <div class="kpi-label">Airlines</div><div class="kpi-value">{df['Airline'].nunique()}</div><div class="kpi-sub">Unique carriers</div></div>
      <div class="kpi-card"><div class="kpi-accent" style="background:#a78bfa"></div>
        <div class="kpi-label">Avg Price</div><div class="kpi-value">₹{df['Price'].mean():,.0f}</div><div class="kpi-sub">Mean fare</div></div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["  🗂️ Preview  ","  📋 Schema  ","  📊 Statistics  ","  🔍 Missing Values  ","  📈 Distributions  "])

    with tabs[0]:
        n = st.slider("Rows to preview", 5, 100, 10)
        st.dataframe(df.head(n), use_container_width=True, height=420)
        if st.checkbox("Show full raw data"):
            st.dataframe(df, use_container_width=True, height=500)

    with tabs[1]:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00d4ff"></span>Column Schema</div>', unsafe_allow_html=True)
        schema = pd.DataFrame([{
            "Column": c, "Data Type": str(df[c].dtype),
            "Unique Values": df[c].nunique(),
            "Null Count": df[c].isnull().sum(),
            "Sample": str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] else "—"
        } for c in df.columns])
        st.dataframe(schema, use_container_width=True, height=420)

    with tabs[2]:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00e5a0"></span>Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.select_dtypes(include=[np.number]).describe().round(2), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#ff6b35"></span>Categorical Value Counts</div>', unsafe_allow_html=True)
        cat_col = st.selectbox("Select column", df.select_dtypes(include="object").columns.tolist())
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, "Count"]
        vc["Pct %"] = (vc["Count"] / len(df) * 100).round(2)
        st.dataframe(vc, use_container_width=True, height=320)

    with tabs[3]:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#f5c842"></span>Missing Value Analysis</div>', unsafe_allow_html=True)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column","Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
        miss["Status"] = miss["Missing %"].apply(lambda x: "✅ Clean" if x == 0 else f"⚠️ {x:.1f}%")
        st.dataframe(miss, use_container_width=True, height=360)
        if miss["Missing Count"].sum() == 0:
            st.success("✅ No missing values found!")

    with tabs[4]:
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="sec-title" style="font-size:1rem"><span class="sec-dot" style="background:#00d4ff"></span>Price Distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(df["Price"], bins=40, color=CYAN, edgecolor="#07090f", linewidth=0.4, alpha=0.9)
            ax.axvline(df["Price"].mean(),   color=ORANGE, lw=1.8, ls="--", label=f"Mean ₹{df['Price'].mean():,.0f}")
            ax.axvline(df["Price"].median(), color=GREEN,  lw=1.8, ls="--", label=f"Median ₹{df['Price'].median():,.0f}")
            ax.set_xlabel("Price (₹)", fontsize=8); ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7.5, framealpha=0)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with cb:
            st.markdown('<div class="sec-title" style="font-size:1rem"><span class="sec-dot" style="background:#ff6b35"></span>Flights by Airline</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
            ac = df["Airline"].value_counts()
            bc2 = [CYAN if i == 0 else DIM for i in range(len(ac))]
            ax2.barh(ac.index[::-1], ac.values[::-1], color=bc2[::-1], height=0.65, edgecolor="#07090f", lw=0.3)
            for val, bar in zip(ac.values[::-1], ax2.patches):
                ax2.text(val+30, bar.get_y()+bar.get_height()/2, f"{val:,}", va="center", fontsize=7, color="#6a7a92")
            ax2.set_xlabel("Count", fontsize=8); ax2.grid(axis="x", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        cc, cd = st.columns(2)
        with cc:
            st.markdown('<div class="sec-title" style="font-size:1rem"><span class="sec-dot" style="background:#00e5a0"></span>Stops Distribution</div>', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
            sv = df["Total_Stops"].value_counts()
            ax3.bar(sv.index, sv.values, color=[GREEN,CYAN,ORANGE,GOLD,PURPLE][:len(sv)],
                    edgecolor="#07090f", lw=0.4, width=0.6)
            for bar, val in zip(ax3.patches, sv.values):
                ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+40, f"{val:,}", ha="center", fontsize=8, color="#6a7a92")
            ax3.set_ylabel("Count", fontsize=8); ax3.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=20, ha="right"); plt.tight_layout(); st.pyplot(fig3); plt.close()

        with cd:
            st.markdown('<div class="sec-title" style="font-size:1rem"><span class="sec-dot" style="background:#f5c842"></span>Source City Share</div>', unsafe_allow_html=True)
            fig4, ax4 = plt.subplots(figsize=(5.5, 3.5))
            src = df["Source"].value_counts()
            w, t, at = ax4.pie(src.values, labels=src.index, autopct="%1.1f%%",
                               colors=[CYAN,ORANGE,GREEN,GOLD,PURPLE], startangle=90,
                               pctdistance=0.78, wedgeprops=dict(linewidth=2, edgecolor="#0d1117"))
            for tx in t: tx.set_fontsize(8); tx.set_color("#8a9ab2")
            for a in at: a.set_fontsize(7); a.set_color("#07090f"); a.set_fontweight("bold")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

# ══════════════════════════════════════════════
#  PAGE: MODEL PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮  Model Prediction":

    bundle = load_model_bundle()
    if bundle is None:
        st.error("⚠️ Model and dataset not found. Place flight_price_model.pkl (or Data_Train.xlsx) next to app.py.")
        st.stop()

    model      = bundle["model"]
    le_airline = bundle["le_airline"]
    le_source  = bundle["le_source"]
    le_dest    = bundle["le_dest"]
    # Always use the canonical FEATURE_COLS — never trust a stale pkl key
    feat_cols  = FEATURE_COLS

    # ── INPUT FORM ──────────────────────────────
    st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00d4ff"></span>Enter Flight Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-form">', unsafe_allow_html=True)

    st.markdown('<span class="form-section-label">✦ Flight Identifiers</span>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: airline = st.selectbox("✈️ Airline",      AIRLINES,     index=3)
    with col2: source  = st.selectbox("🛫 Source City",  SOURCES,      index=2)
    with col3: dest    = st.selectbox("🛬 Destination",  DESTINATIONS, index=5)
    with col4: stops   = st.selectbox("🔁 Total Stops",  STOPS_OPTS,   index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="form-section-label">✦ Journey Timing</span>', unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5: journey_date = st.date_input("📅 Journey Date",     value=date(2024, 5, 15))
    with col6: dep_time     = st.time_input("🕐 Departure Time",   value=time(8, 30))
    with col7: arr_time     = st.time_input("🕓 Arrival Time",     value=time(11, 45))
    with col8: dur_hrs      = st.number_input("⏱️ Duration (hours)", min_value=0.5,
                                               max_value=24.0, value=3.25, step=0.25)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btnc, _ = st.columns([1, 1.5, 1])
    with btnc:
        clicked = st.button("🔮  PREDICT FLIGHT PRICE", use_container_width=True)

    if clicked:
        try:
            dur_mins = int(dur_hrs * 60)

            # Build input DataFrame — uses FEATURE_COLS strictly
            X_in = build_input_row(
                airline, source, dest, stops,
                journey_date, dep_time, arr_time, dur_mins,
                le_airline, le_source, le_dest,
            )

            price = max(float(model.predict(X_in)[0]), 0)

            st.markdown(f"""
            <div class="result-box">
              <div class="result-label">Estimated Flight Price</div>
              <div class="result-price">₹ {price:,.0f}</div>
              <div class="result-note">
                {airline} &nbsp;·&nbsp; {source} → {dest} &nbsp;·&nbsp;
                {stops} &nbsp;·&nbsp; {journey_date.strftime('%d %b %Y')}
              </div>
              <div class="result-conf">
                🤖 Random Forest Regressor &nbsp;|&nbsp;
                R² = <b style='color:#00d4ff'>0.9490</b> &nbsp;|&nbsp;
                Duration: {dur_mins // 60}h {dur_mins % 60}m
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature summary table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00e5a0"></span>Input Feature Summary</div>', unsafe_allow_html=True)
            stops_num = int(stops.replace("non-stop","0").split()[0])
            summary = pd.DataFrame({
                "Feature":  ["Airline","Source","Destination","Total Stops",
                             "Journey Day","Journey Month","Dep Hour","Dep Min",
                             "Arr Hour","Arr Min","Duration (mins)"],
                "Raw Value":[airline, source, dest, stops,
                             journey_date.day, journey_date.month,
                             dep_time.hour, dep_time.minute,
                             arr_time.hour, arr_time.minute, dur_mins],
                "Model Input": X_in.values[0].tolist(),
            })
            st.dataframe(summary, use_container_width=True, hide_index=True, height=380)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("Tip: Make sure flight_price_model.pkl is the one generated alongside this app.py.")

    # ── ANALYTICS CHARTS ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#f5c842"></span>Price Analytics</div>', unsafe_allow_html=True)

    df = load_data()
    if df is not None:
        dc = df.dropna().copy()
        v1, v2, v3 = st.columns(3)

        with v1:
            st.markdown("**✈️ Airline vs Avg Price**")
            aa = dc.groupby("Airline")["Price"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 4.5))
            bca = [CYAN if i == len(aa)-1 else DIM for i in range(len(aa))]
            ax.barh(aa.index, aa.values, color=bca, height=0.65, edgecolor="#07090f", lw=0.3)
            for val, bar in zip(aa.values, ax.patches):
                ax.text(val+200, bar.get_y()+bar.get_height()/2, f"₹{val:,.0f}", va="center", fontsize=6.5, color="#6a7a92")
            ax.set_xlabel("Avg Price (₹)", fontsize=7.5)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with v2:
            st.markdown("**🏙️ Source vs Avg Price**")
            sa = dc.groupby("Source")["Price"].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(5, 4.5))
            bars2 = ax2.bar(sa.index, sa.values, color=[CYAN,ORANGE,GREEN,GOLD,PURPLE][:len(sa)],
                            edgecolor="#07090f", lw=0.4, width=0.6)
            for bar, val in zip(bars2, sa.values):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80, f"₹{val:,.0f}",
                         ha="center", fontsize=7, color="#6a7a92")
            ax2.set_ylabel("Avg Price (₹)", fontsize=7.5)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax2.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        with v3:
            st.markdown("**🔁 Stops vs Avg Price**")
            ds = dc.copy()
            ds["Total_Stops"] = ds["Total_Stops"].replace("non-stop","0 stop")
            ds["SN"] = ds["Total_Stops"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
            spa = ds.groupby("SN")["Price"].mean().sort_index()
            labels3 = ["Non-stop" if s == 0 else f"{s} stop{'s' if s>1 else ''}" for s in spa.index]
            fig3, ax3 = plt.subplots(figsize=(5, 4.5))
            bars3 = ax3.bar(labels3, spa.values, color=[GREEN,CYAN,ORANGE,GOLD,PURPLE][:len(spa)],
                            edgecolor="#07090f", lw=0.4, width=0.6)
            for bar, val in zip(bars3, spa.values):
                ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80, f"₹{val:,.0f}",
                         ha="center", fontsize=7.5, color="#6a7a92")
            ax3.set_ylabel("Avg Price (₹)", fontsize=7.5)
            ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax3.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        # Actual vs Predicted + Model R² comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#a78bfa"></span>Model Performance</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)

        actuals  = [16655,4959,9187,3858,12898,10529,16079,7229,10844,16289,3943,8371,14815,3841,5126,2754,4174,6171,15809,8452,2175,4823,7608,5613,15812,5406,3100,13014,4995,8040]
        preds_rf = [16850,5497,8861,3669,12898,9573,16081,7229,10470,16267,4103,8866,16311,3850,6385,3053,4211,6165,15407,8552,2121,4796,6183,5631,15740,5646,3282,13356,4995,6281]

        with p1:
            fig4, ax4 = plt.subplots(figsize=(6, 4.2))
            ax4.scatter(actuals, preds_rf, color=CYAN, alpha=0.75, s=45, edgecolors="#07090f", linewidths=0.5)
            mn, mx = min(actuals), max(actuals)
            ax4.plot([mn,mx],[mn,mx], color=ORANGE, lw=1.5, ls="--", label="Perfect Fit")
            ax4.set_xlabel("Actual Price (₹)", fontsize=8)
            ax4.set_ylabel("Predicted Price (₹)", fontsize=8)
            ax4.set_title("Actual vs Predicted — Random Forest", fontsize=9, color="#dde4ef")
            ax4.legend(fontsize=7.5, framealpha=0)
            ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{int(x/1000)}K"))
            ax4.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        with p2:
            fig5, ax5 = plt.subplots(figsize=(6, 4.2))
            mnames = ["Linear\nRegression","Decision\nTree","KNN","Random\nForest","SVR"]
            r2s    = [0.6190, 0.9204, 0.7534, 0.9490, 0.3863]
            bc5 = [CYAN if v == max(r2s) else DIM for v in r2s]
            bars5 = ax5.bar(mnames, r2s, color=bc5, edgecolor="#07090f", lw=0.4, width=0.55)
            for bar, val in zip(bars5, r2s):
                ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                         f"{val:.4f}", ha="center", fontsize=8, color="#6a7a92")
            ax5.set_ylabel("R² Score", fontsize=8)
            ax5.set_title("All Models — R² Comparison", fontsize=9, color="#dde4ef")
            ax5.set_ylim(0, 1.1); ax5.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig5); plt.close()

# ══════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════
elif page == "ℹ️  About":

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00d4ff"></span>Project Description</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:.9rem;line-height:1.9;color:#8a9ab2'>
        <b style='color:#dde4ef'>SkyFare Intelligence</b> predicts Indian domestic flight fares using ML.
        The dataset has <b style='color:#dde4ef'>10,683 flights</b> across 12 airlines, 5 sources and 6 destinations,
        with fares ₹1,759–₹79,512.<br><br>
        Five regression models were evaluated. <b style='color:#00d4ff'>Random Forest</b> performed best with
        R²=<b style='color:#00d4ff'>0.9490</b> and MAE=<b style='color:#00d4ff'>0.0655</b>.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#ff6b35"></span>Model Comparison</div>', unsafe_allow_html=True)
        results = pd.DataFrame({
            "Model":    ["Random Forest ★","Decision Tree","KNN","Linear Regression","SVR"],
            "R² Score": [0.9490, 0.9204, 0.7534, 0.6190, 0.3863],
            "MAE":      [0.0655, 0.0694, 0.1758, 0.2558, 0.3287],
            "MSE":      [0.0139, 0.0217, 0.0671, 0.1037, 0.1670],
        })
        st.dataframe(results.style.format({"R² Score":"{:.4f}","MAE":"{:.4f}","MSE":"{:.4f}"}),
                     use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00e5a0"></span>Features Used by Model</div>', unsafe_allow_html=True)
        feat_desc = {
            "Airline":       "Label-encoded carrier name",
            "Source":        "Label-encoded source city",
            "Destination":   "Label-encoded destination",
            "Total_Stops":   "Integer stop count (non-stop → 0)",
            "Journey_Day":   "Day of month (1–31)",
            "Journey_Month": "Month number (1–12)",
            "Dep_hour":      "Departure hour (0–23)",
            "Dep_min":       "Departure minute (0–59)",
            "Arr_hour":      "Arrival hour (0–23)",
            "Arr_min":       "Arrival minute (0–59)",
            "Duration_mins": "Total flight duration in minutes",
        }
        for k, v in feat_desc.items():
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:.4rem 0;
                        border-bottom:1px solid #1c2a3a;font-size:.78rem'>
              <span style='color:#00d4ff;font-weight:600;font-family:monospace'>{k}</span>
              <span style='color:#6a7a92'>{v}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#00e5a0"></span>Tech Stack</div>', unsafe_allow_html=True)
        tools = [("🐍","Python 3.10+"),("📊","Streamlit"),("🤖","Scikit-learn"),("🌲","Random Forest"),
                 ("🐼","Pandas"),("🔢","NumPy"),("📈","Matplotlib"),("📦","Pickle"),
                 ("📋","OpenPyXL"),("🧪","Train/Test Split"),("🏷️","LabelEncoder"),("⚙️","Feature Eng.")]
        st.markdown('<div class="tools-grid">', unsafe_allow_html=True)
        for icon, name in tools:
            st.markdown(f'<div class="tool-pill"><span>{icon}</span>{name}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="sec-dot" style="background:#f5c842"></span>Author</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#0d1f35;border:1px solid #1c2a3a;border-radius:12px;padding:1.5rem;text-align:center'>
          <div style='font-size:2.5rem;margin-bottom:.5rem'>👤</div>
          <div style='font-family:Bebas Neue,sans-serif;font-size:1.3rem;letter-spacing:.08em;color:#fff'>
            Data Science Student</div>
          <div style='font-size:.76rem;color:#4a5a72;margin-top:.3rem'>
            Built with ❤️ using Python & Streamlit</div>
          <div style='margin-top:1rem;display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap'>
            <span style='background:#1c2a3a;border-radius:99px;padding:.3rem .8rem;font-size:.7rem;color:#8a9ab2'>Machine Learning</span>
            <span style='background:#1c2a3a;border-radius:99px;padding:.3rem .8rem;font-size:.7rem;color:#8a9ab2'>Data Science</span>
            <span style='background:#1c2a3a;border-radius:99px;padding:.3rem .8rem;font-size:.7rem;color:#8a9ab2'>Python</span>
            <span style='background:#1c2a3a;border-radius:99px;padding:.3rem .8rem;font-size:.7rem;color:#8a9ab2'>Streamlit</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1c2a3a;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem'>
  <div style='font-family:Bebas Neue,sans-serif;font-size:1rem;letter-spacing:.06em;color:#1c2a3a'>
    SKY<span style='color:#1c3a55'>FARE</span> Intelligence
  </div>
  <div style='font-size:.7rem;color:#1c2a3a'>
    Streamlit · Scikit-learn · Random Forest · R²=0.9490
  </div>
</div>
""", unsafe_allow_html=True)
