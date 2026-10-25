"""Flight Price Predictor - Streamlit app (redesigned UI).

Run:   streamlit run app.py
Needs: streamlit>=1.40, pandas, scikit-learn  (altair ships with streamlit)
Data:  DATASET_clean.csv in the folder you run from, columns:
       class (0=Economy, 1=Business), duration, days_left, stops, price
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

FEATURES = ["class", "duration", "days_left", "stops"]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

:root{
  --bg:#0b1020; --card:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.12);
  --txt:#e8ecff; --muted:#9aa3c7; --violet:#7c5cff; --cyan:#22d3ee;
  --good:#34d399; --bad:#fb7185;
}

/* page */
.stApp{
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(34,211,238,.18), transparent 55%),
    var(--bg);
  color:var(--txt);
  font-family:'Inter',system-ui,sans-serif;
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton{display:none;}
.block-container{max-width:1150px; padding-top:2.4rem; padding-bottom:3rem;}
[data-testid="stWidgetLabel"] p{color:var(--txt); font-weight:500;}

/* hero */
.eyebrow{display:inline-block; padding:.3rem .85rem; border-radius:999px; font-size:.72rem;
  letter-spacing:.16em; font-weight:600; color:var(--cyan);
  background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.3);}
.hero-title{font-family:'Space Grotesk',sans-serif; font-size:3.2rem; font-weight:700;
  line-height:1.05; margin:.9rem 0 .5rem;
  background:linear-gradient(90deg,#fff,#b9c4ff 55%,var(--cyan));
  -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero-sub{color:var(--muted); font-size:1.05rem; max-width:46rem; margin-bottom:1.6rem;}

/* input panel (st.container(key="inputs")) */
.st-key-inputs{background:var(--card); border:1px solid var(--stroke); border-radius:22px;
  padding:1.4rem 1.6rem 1rem; backdrop-filter:blur(10px);}
.panel-title{font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; margin-bottom:.6rem;}

/* boarding-pass result card */
.ticket{position:relative; border-radius:24px; padding:1.5rem 1.7rem 1.2rem;
  background:linear-gradient(160deg, rgba(124,92,255,.24), rgba(34,211,238,.08) 60%, rgba(255,255,255,.04));
  border:1px solid var(--stroke);
  box-shadow:0 20px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);}
.t-head{display:flex; justify-content:space-between; align-items:center;}
.chip{padding:.25rem .8rem; border-radius:999px; font-size:.72rem; font-weight:700;
  letter-spacing:.12em; text-transform:uppercase; color:#fff;
  background:linear-gradient(90deg,var(--violet),var(--cyan));}
.t-id{font-size:.7rem; letter-spacing:.14em; color:var(--muted);}
.t-path{display:flex; align-items:center; gap:.6rem; margin:1.4rem 0 .4rem;}
.t-path .dot{width:10px; height:10px; border-radius:50%; border:2px solid var(--cyan);}
.t-path .line{flex:1; height:0; border-top:2px dashed rgba(255,255,255,.28); position:relative;}
.t-path .line i{position:absolute; left:50%; top:-15px; transform:translateX(-50%);
  font-style:normal; font-size:1.3rem; background:#161a3a; padding:0 .45rem;}
.t-meta{display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted);}
.t-label{margin-top:1.4rem; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted);}
.t-price{font-family:'Space Grotesk',sans-serif; font-size:3.8rem; font-weight:700; line-height:1.05;
  background:linear-gradient(90deg,#fff,#c7d2ff 55%,var(--cyan));
  -webkit-background-clip:text; background-clip:text; color:transparent;}
.t-range{margin-top:.25rem; color:var(--muted); font-size:.85rem;}
.t-tear{position:relative; margin:1.3rem -1.7rem 1.1rem; border-top:2px dashed rgba(255,255,255,.18);}
.t-tear:before,.t-tear:after{content:""; position:absolute; top:-12px; width:22px; height:22px;
  border-radius:50%; background:var(--bg); border:1px solid var(--stroke);}
.t-tear:before{left:-11px;}
.t-tear:after{right:-11px;}
.t-foot{display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem;}
.t-foot small{display:block; color:var(--muted); font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;}
.t-foot b{font-size:1rem; font-weight:600;}
.up{color:var(--bad);} .down{color:var(--good);}
.barcode{height:32px; margin-top:1.1rem; opacity:.5; border-radius:4px;
  background:repeating-linear-gradient(90deg,#fff 0 2px,transparent 2px 5px,#fff 5px 6px,transparent 6px 10px,#fff 10px 13px,transparent 13px 15px);}

/* class comparison cards */
.minis{display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin-top:1rem;}
.mini{padding:.8rem 1rem; border-radius:16px; background:var(--card); border:1px solid var(--stroke);}
.mini.on{border-color:var(--cyan); box-shadow:0 0 0 1px rgba(34,211,238,.35), 0 8px 30px rgba(34,211,238,.12);}
.mini small{color:var(--muted); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;}
.mini b{display:block; font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700;}

/* insights */
.sec{font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; margin:2.2rem 0 .4rem;}
.tip{margin-top:.6rem; padding:.8rem 1rem; border-radius:14px; font-size:.92rem;
  background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.3);}
.drivers{display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.6rem 0 .8rem;}
.driver{padding:1rem 1.1rem; border-radius:16px; background:var(--card); border:1px solid var(--stroke);}
.driver small{display:block; color:var(--muted); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;}
.driver b{display:block; font-family:'Space Grotesk',sans-serif; font-size:1.5rem; margin:.25rem 0 .1rem;}
.driver span{color:var(--muted); font-size:.8rem;}
.pills{display:flex; gap:.6rem; flex-wrap:wrap; margin-top:1.8rem;}
.pill{padding:.45rem .9rem; border-radius:999px; font-size:.8rem; color:var(--muted);
  background:var(--card); border:1px solid var(--stroke);}
.pill b{color:var(--txt);}
.fine{margin-top:.8rem; color:var(--muted); font-size:.78rem;}

@media (max-width: 800px){
  .drivers{grid-template-columns:repeat(2,1fr);}
  .t-price{font-size:3rem;} .hero-title{font-size:2.3rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def inr(x: float) -> str:
    """Format as Indian rupees with lakh/crore grouping: 1234567 -> ₹12,34,567."""
    n = int(round(x))
    sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join(reversed([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)]))
        s = f"{head},{tail}"
    return f"{sign}₹{s}"


def html(s: str) -> str:
    """Strip indentation/newlines so Markdown never turns the HTML into a code block."""
    return " ".join(line.strip() for line in s.splitlines())


def stops_label(s: int) -> str:
    return "Non-stop" if s == 0 else ("1 stop" if s == 1 else f"{int(s)} stops")


# ----------------------------------------------------------------------------
# Model (trained once, cached - not on every widget change)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model...")
def train():
    df = pd.read_csv("DATASET_clean.csv")
    missing = [c for c in FEATURES + ["price"] if c not in df.columns]
    if missing:
        raise ValueError(f"DATASET_clean.csv is missing columns: {', '.join(missing)}")

    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    test_pred = model.predict(X_test)

    return {
        "model": model,
        "r2": float(r2_score(y_test, test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mean_price": float(y.mean()),
        "dur_min": float(np.floor(df["duration"].min())),
        "dur_max": float(np.ceil(df["duration"].max())),
        "day_min": int(df["days_left"].min()),
        "day_max": int(df["days_left"].max()),
        "stop_opts": sorted(int(s) for s in df["stops"].unique()),
        "coef": dict(zip(FEATURES, (float(c) for c in model.coef_))),
    }


try:
    M = train()
except FileNotFoundError:
    st.error("`DATASET_clean.csv` not found. Put it in the folder you run `streamlit run` from.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

model = M["model"]


def predict(cls: int, duration: float, days_left: int, stops: int) -> float:
    row = pd.DataFrame([{"class": cls, "duration": duration, "days_left": days_left, "stops": stops}])
    return max(float(model.predict(row[FEATURES])[0]), 0.0)


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    html("""
    <div class="eyebrow">✈ MACHINE LEARNING · LINEAR REGRESSION</div>
    <div class="hero-title">Flight Price Predictor</div>
    <div class="hero-sub">Dial in your trip and get an instant fare estimate, plus how your
    booking choices move the price.</div>
    """),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Inputs (left) + result (right)
# ----------------------------------------------------------------------------
left, right = st.columns([5, 6], gap="large")

with left:
    with st.container(key="inputs"):
        st.markdown('<div class="panel-title">Plan your trip</div>', unsafe_allow_html=True)

        cabin = st.segmented_control("Cabin class", ["Economy", "Business"], default="Economy") or "Economy"

        duration = st.slider(
            "Flight duration (hours)",
            min_value=M["dur_min"], max_value=M["dur_max"],
            value=float(min(max(10.0, M["dur_min"]), M["dur_max"])), step=0.5,
        )
        days_left = st.slider(
            "Days left before departure",
            min_value=M["day_min"], max_value=M["day_max"],
            value=int(min(max(20, M["day_min"]), M["day_max"])),
        )

        default_stops = 1 if 1 in M["stop_opts"] else M["stop_opts"][0]
        stops = st.segmented_control(
            "Number of stops", M["stop_opts"], default=default_stops, format_func=stops_label
        )
        if stops is None:
            stops = default_stops

cls = 0 if cabin == "Economy" else 1
price = predict(cls, duration, days_left, stops)
lo, hi = max(price - M["rmse"], 0.0), price + M["rmse"]
econ = predict(0, duration, days_left, stops)
biz = predict(1, duration, days_left, stops)
delta = price / M["mean_price"] - 1 if M["mean_price"] else 0.0

with right:
    st.markdown(
        html(f"""
        <div class="ticket">
          <div class="t-head"><span class="chip">{cabin}</span><span class="t-id">FARE ESTIMATE</span></div>
          <div class="t-path"><span class="dot"></span><span class="line"><i>✈</i></span><span class="dot"></span></div>
          <div class="t-meta"><span>{duration:g} h flight</span><span>{stops_label(stops)}</span></div>
          <div class="t-label">Predicted fare</div>
          <div class="t-price">{inr(price)}</div>
          <div class="t-range">Typical range {inr(lo)} – {inr(hi)}</div>
          <div class="t-tear"></div>
          <div class="t-foot">
            <div><small>Booked</small><b>{days_left} days ahead</b></div>
            <div><small>Stops</small><b>{stops_label(stops)}</b></div>
            <div><small>Vs avg fare</small>
              <b class="{'up' if delta >= 0 else 'down'}">{'▲' if delta >= 0 else '▼'} {abs(delta) * 100:.0f}%</b></div>
          </div>
          <div class="barcode"></div>
        </div>
        <div class="minis">
          <div class="mini {'on' if cls == 0 else ''}"><small>Economy</small><b>{inr(econ)}</b></div>
          <div class="mini {'on' if cls == 1 else ''}"><small>Business</small><b>{inr(biz)}</b></div>
          <div class="mini"><small>Business premium</small><b>{(biz / econ):.1f}×</b></div>
        </div>
        """) if econ > 0 else html(f"""
        <div class="ticket">
          <div class="t-head"><span class="chip">{cabin}</span><span class="t-id">FARE ESTIMATE</span></div>
          <div class="t-label">Predicted fare</div>
          <div class="t-price">{inr(price)}</div>
          <div class="t-range">Typical range {inr(lo)} – {inr(hi)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Insights
# ----------------------------------------------------------------------------
st.markdown('<div class="sec">Insights</div>', unsafe_allow_html=True)
tab_window, tab_drivers = st.tabs(["📈 Booking window", "🧭 What moves the price"])

with tab_window:
    days = np.arange(M["day_min"], M["day_max"] + 1)
    curve = pd.DataFrame({"class": cls, "duration": duration, "days_left": days, "stops": stops})
    curve["price"] = np.clip(model.predict(curve[FEATURES]), 0, None)
    now = pd.DataFrame({"days_left": [days_left], "price": [price]})

    base = alt.Chart(curve).encode(
        x=alt.X("days_left:Q", title="Days left before departure  →  closer to takeoff",
                scale=alt.Scale(reverse=True, nice=False)),
        y=alt.Y("price:Q", title="Predicted fare (₹)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("days_left:Q", title="Days left"),
                 alt.Tooltip("price:Q", title="Fare (₹)", format=",.0f")],
    )
    fill = alt.Gradient(
        gradient="linear", x1=1, x2=1, y1=1, y2=0,
        stops=[alt.GradientStop(color="rgba(124,92,255,0)", offset=0),
               alt.GradientStop(color="rgba(124,92,255,0.45)", offset=1)],
    )
    area = base.mark_area(color=fill)
    line = base.mark_line(color="#7c5cff", strokeWidth=3)
    rule = alt.Chart(now).mark_rule(color="#22d3ee", strokeDash=[4, 4], opacity=0.7).encode(x="days_left:Q")
    dot = alt.Chart(now).mark_point(size=190, filled=True, color="#22d3ee", stroke="white", strokeWidth=2)\
        .encode(x="days_left:Q", y="price:Q")

    chart = (
        (area + line + rule + dot)
        .properties(height=300)
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#9aa3c7", titleColor="#9aa3c7", gridColor="rgba(255,255,255,.07)",
                        domainColor="rgba(255,255,255,.15)", tickColor="rgba(255,255,255,.15)")
    )
    st.altair_chart(chart)

    best = curve.loc[curve["price"].idxmin()]
    best_day, saving = int(best["days_left"]), price - float(best["price"])
    if saving > 1 and best_day > days_left:
        tip = f"Booking <b>{best_day - days_left} days earlier</b> ({best_day} days out) could save about <b>{inr(saving)}</b>."
    elif saving > 1:
        tip = f"Waiting until <b>{best_day} days</b> before departure could save about <b>{inr(saving)}</b>."
    else:
        tip = "You're already at the cheapest point of the booking window for this trip."
    st.markdown(f'<div class="tip">💡 {tip}</div>', unsafe_allow_html=True)

with tab_drivers:
    c = M["coef"]
    drivers = [
        ("Business vs Economy", c["class"], "cabin upgrade"),
        ("Each extra flight hour", c["duration"], "longer journey"),
        ("Each extra day booked ahead", c["days_left"], "earlier booking"),
        ("Each extra stop", c["stops"], "more connections"),
    ]
    cards = "".join(
        f'<div class="driver"><small>{label}</small>'
        f'<b class="{"up" if v >= 0 else "down"}">{"▲" if v >= 0 else "▼"} {inr(abs(v))}</b>'
        f'<span>{note} · fare goes {"up" if v >= 0 else "down"}</span></div>'
        for label, v, note in drivers
    )
    st.markdown(f'<div class="drivers">{cards}</div>', unsafe_allow_html=True)
    st.caption("Effect on the predicted fare of changing one input while the others stay the same.")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    html(f"""
    <div class="pills">
      <span class="pill">Model <b>Linear Regression</b></span>
      <span class="pill">R² (test) <b>{M['r2']:.3f}</b></span>
      <span class="pill">Typical error <b>±{inr(M['rmse'])}</b></span>
      <span class="pill">Trained on <b>{M['n_train']:,}</b> flights</span>
    </div>
    <div class="fine">Estimates only, not an actual fare quote. Inputs are limited to the range of the training data.</div>
    """),
    unsafe_allow_html=True,
)