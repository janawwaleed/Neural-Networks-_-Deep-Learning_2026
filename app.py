import base64
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from adaline import test_adaline, train_adaline
from preprocessing import load_data, split_data
from SLP import Perceptron

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
APP_TITLE = "Neural Networks Task1"
LOGO_PATH = Path(__file__).parent / "brain_logo.png"

FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
CLASSES = ["Adelie", "Chinstrap", "Gentoo"]

ALGO_TITLES = {"perceptron": "Single Layer Perceptron", "adaline": "Adaline"}

# colour theme: (gradient start, gradient end) per algorithm page
ACCENTS = {
    "perceptron": ("#E07B7B", "#F7AAA2"),
    "adaline": ("#E07B7B", "#F7AAA2"),
}
LANDING_ACCENT = ACCENTS["perceptron"]

INK = "#2B2D42"
CLASS_COLORS = ("#E4605E", "#2F6F9F")  # class 1 (label -1), class 2 (label +1)
CLASS_LIGHT = ("#F7C9C8", "#BCD6EA")


# --------------------------------------------------------------------------
# Glue: connects the GUI to preprocessing.py, SLP.py and adaline.py
# --------------------------------------------------------------------------
@st.cache_data
def _load_df():
    return load_data()



def train_and_evaluate(algo: str, cfg: dict) -> dict:
    df = _load_df()
    X_tr, y_tr, X_te, y_te, mean, std = split_data(
        df, cfg["class_1"], cfg["class_2"], cfg["feature_1"], cfg["feature_2"], return_stats=True)

    # preprocessing labels class 1 as +1, but the GUI treats class 1 as -1 -> flip
    y_tr, y_te = -y_tr, -y_te

    if algo == "perceptron":
        model = Perceptron(cfg["eta"], cfg["epochs"], cfg["use_bias"])
        model.train(X_tr, y_tr)
        conf, acc = model.test(X_te, y_te)
        if cfg["use_bias"]:                      # w = [w0, w1, w2]
            b_z, w_z = model.w[0], model.w[1:]
        else:                                    # w = [w1, w2]
            b_z, w_z = 0.0, model.w
        history = []
    else:
        w_z, b_z, history = train_adaline(
            X_tr, y_tr, cfg["eta"], cfg["epochs"], cfg["mse_threshold"], cfg["use_bias"])
        model = None
        _, conf, acc = test_adaline(X_te, y_te, w_z, b_z)

    # weights were learned on standardized features z = (x - mean) / std.
    # Convert to original units so the line is drawn correctly:
    w1 = w_z[0] / std[0]
    w2 = w_z[1] / std[1]
    b = b_z - w_z[0] * mean[0] / std[0] - w_z[1] * mean[1] / std[1]

    return {
        "x_train": X_tr * std + mean, "y_train": y_tr,     # back to original units for the plot
        "x_test": X_te * std + mean, "y_test": y_te,
        "w1": float(w1), "w2": float(w2), "b": float(b),
        "history": list(history),
        "confusion": np.asarray(conf),
        "accuracy": float(acc),
        "model": model,
    }


def classify_sample(result: dict, cfg: dict, v1: float, v2: float) -> str:
    """Uses the weights in original feature units, so v1 and v2 are typed as-is."""
    score = result["w1"] * v1 + result["w2"] * v2 + result["b"]
    return cfg["class_2"] if score >= 0 else cfg["class_1"]


# --------------------------------------------------------------------------
# Navigation helpers
# --------------------------------------------------------------------------
def go(page):
    """Switch page and clear any previous results."""
    st.session_state["page"] = page
    st.session_state.pop("result", None)
    st.session_state.pop("cfg", None)


# --------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------
def html(markup: str):
    """Render raw HTML. Leading spaces / blank lines would make Streamlit's
    markdown treat the block as code, so strip them first."""
    clean = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


@st.cache_data
def logo_base64() -> str:
    """The logo as base64 (empty string if the file is missing)."""
    try:
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:
        return ""


def logo_img(height: int) -> str:
    data = logo_base64()
    if data:
        return f'<img src="data:image/png;base64,{data}" style="height:{height}px;" alt="logo"/>'
    return f'<span style="font-size:{int(height * 0.8)}px;">🧠</span>'


CSS = """
<style>
:root{--a1:__A1__;--a2:__A2__;--ink:#2B2D42;--muted:#6B6F85;}
.stApp{background:linear-gradient(180deg,#FFF7F6 0%,#FBEEF3 100%);}
#MainMenu,footer,.stDeployButton,[data-testid="stAppDeployButton"]{visibility:hidden;display:none;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;max-width:1400px;}
.stApp h1,.stApp h2,.stApp h3,.stApp label,.stApp [data-testid="stWidgetLabel"] p{color:var(--ink) !important;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF 0%,#FDEFEF 100%);border-right:1px solid #F2D7D7;}
.side-logo{display:flex;align-items:center;gap:10px;padding:4px 0 14px 0;border-bottom:1px solid #F2D7D7;margin-bottom:10px;}
.side-logo span{font-weight:800;color:var(--ink);line-height:1.15;font-size:15px;}
.side-section{margin:16px 0 4px 0;padding:4px 10px;border-radius:8px;font-size:12px;font-weight:700;
letter-spacing:.08em;text-transform:uppercase;color:#fff;background:linear-gradient(90deg,var(--a1),var(--a2));}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;display:flex;align-items:center;gap:26px;padding:26px 32px;border-radius:22px;
background:linear-gradient(120deg,var(--a1) 0%,var(--a2) 100%);box-shadow:0 14px 34px -14px var(--a1);margin-bottom:22px;}
.hero::before{content:"";position:absolute;right:-60px;top:-70px;width:260px;height:260px;border-radius:50%;background:rgba(255,255,255,.14);}
.hero::after{content:"";position:absolute;right:120px;bottom:-90px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,.10);}
.hero.small{padding:14px 24px;gap:18px;margin-bottom:16px;}
.logo-badge{flex:0 0 auto;display:flex;align-items:center;justify-content:center;width:118px;height:118px;border-radius:50%;
background:#fff;box-shadow:0 8px 22px -8px rgba(0,0,0,.35);z-index:1;}
.hero.small .logo-badge{width:70px;height:70px;}
.hero-text{z-index:1;}
.hero-kicker{font-size:12px;font-weight:700;letter-spacing:.16em;color:rgba(255,255,255,.88);text-transform:uppercase;}
.hero-title{font-size:42px;font-weight:800;line-height:1.1;color:#fff;margin:4px 0 6px 0;text-shadow:0 2px 10px rgba(0,0,0,.12);}
.hero.small .hero-title{font-size:28px;margin:2px 0 2px 0;}
.hero-sub{font-size:16px;color:rgba(255,255,255,.95);}

/* ---------- landing cards ---------- */
.section-title{font-size:20px;font-weight:800;color:var(--ink);margin:6px 0 12px 0;}
.card{position:relative;overflow:hidden;background:#fff;border-radius:20px;padding:26px 26px 22px 26px;
box-shadow:0 10px 30px -14px rgba(43,45,66,.30);border:1px solid #F4E4E4;margin-bottom:10px;}
.card-strip{position:absolute;left:0;top:0;right:0;height:8px;}
.algo-card.perceptron .card-strip{background:linear-gradient(90deg,#E07B7B,#F7AAA2);}
.algo-card.adaline .card-strip{background:linear-gradient(90deg,#E07B7B,#F7AAA2);}
.tag{display:inline-block;font-size:11px;font-weight:800;letter-spacing:.1em;padding:4px 10px;border-radius:999px;color:#fff;}
.algo-card.perceptron .tag{background:#E07B7B;}
.algo-card.adaline .tag{background:#E07B7B;}
.card-title{font-size:25px;font-weight:800;color:var(--ink);margin:10px 0 6px 0;}
.card-text{font-size:15px;color:var(--muted);margin-bottom:12px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
.chip{font-size:12.5px;font-weight:600;padding:4px 11px;border-radius:999px;}
.algo-card.perceptron .chip{background:#FDECEB;color:#B84A4A;}
.algo-card.adaline .chip{background:#FDECEB;color:#B84A4A;}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:6px 0 10px 0;}
.step{background:rgba(255,255,255,.75);border:1px solid #F4E4E4;border-radius:16px;padding:14px 16px;}
.step-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;
background:linear-gradient(120deg,#E07B7B,#F7AAA2);color:#fff;font-weight:800;margin-right:8px;}
.step-title{font-weight:800;color:var(--ink);}
.step-text{font-size:13.5px;color:var(--muted);margin-top:6px;}

/* ---------- results ---------- */
.badge{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:700;font-size:13px;color:#fff;
background:linear-gradient(90deg,var(--a1),var(--a2));}
.metric-row{display:grid;grid-template-columns:1fr;gap:14px;margin:0 0 16px 0;}
.metric{background:#fff;border-radius:16px;padding:14px 18px;border:1px solid #F4E4E4;border-left:6px solid var(--a1);
box-shadow:0 8px 22px -14px rgba(43,45,66,.35);}
.m-label{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.m-value{font-size:30px;font-weight:800;color:var(--ink);line-height:1.2;}
.m-sub{font-size:12.5px;color:var(--muted);}
.panel-title{font-size:16px;font-weight:800;color:var(--ink);margin-bottom:2px;}
.pred-pill{display:inline-block;margin-top:10px;padding:12px 20px;border-radius:14px;color:#fff;font-size:17px;
box-shadow:0 8px 20px -10px rgba(0,0,0,.4);}
.footer-note{text-align:center;color:#B58A8A;font-size:12.5px;margin-top:26px;}

/* ---------- widgets ---------- */
div.stButton > button{border-radius:12px;font-weight:700;border:1px solid #EBCFCF;color:var(--ink);background:#fff;
transition:all .15s ease;padding:.55rem 1rem;}
div.stButton > button:hover{border-color:var(--a1);color:var(--a1);transform:translateY(-1px);}
div.stButton > button[kind="primary"],div.stButton > button[data-testid="stBaseButton-primary"]{
background:linear-gradient(90deg,var(--a1),var(--a2));color:#fff !important;border:none;box-shadow:0 8px 18px -8px var(--a1);}
div.stButton > button[kind="primary"]:hover,div.stButton > button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.06);color:#fff !important;}
.st-key-open_perc div.stButton > button{background:linear-gradient(90deg,#E07B7B,#F7AAA2) !important;color:#fff !important;border:none !important;font-size:16px;box-shadow:0 8px 18px -8px #E07B7B;}
.st-key-open_ada div.stButton > button{background:linear-gradient(90deg,#E07B7B,#F7AAA2) !important;color:#fff !important;border:none !important;font-size:16px;box-shadow:0 8px 18px -8px #E07B7B;}
.st-key-open_perc div.stButton > button:hover,.st-key-open_ada div.stButton > button:hover{color:#fff !important;filter:brightness(1.07);}
.st-key-open_perc div.stButton > button p,.st-key-open_ada div.stButton > button p{color:#fff !important;}
[data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-baseweb="base-input"]{border-radius:10px !important;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px;background:#fff;border-color:#F4E4E4;box-shadow:0 8px 22px -16px rgba(43,45,66,.35);}
@media (max-width:800px){.steps{grid-template-columns:1fr 1fr;}.hero-title{font-size:30px;}}
</style>
"""


def inject_css(accent):
    html(CSS.replace("__A1__", accent[0]).replace("__A2__", accent[1]))


def hero(subtitle: str = "", small: bool = False, title: str = APP_TITLE):
    cls = "hero small" if small else "hero"
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    html(f"""
    <div class="{cls}">
      <div class="logo-badge">{logo_img(46 if small else 88)}</div>
      <div class="hero-text">
        <div class="hero-title">{title}</div>
        {sub_html}
      </div>
    </div>
    """)


def metric_cards(items):
    """items: list of (label, value, sub-text)."""
    cards = "".join(
        f'<div class="metric"><div class="m-label">{l}</div><div class="m-value">{v}</div>'
        f'<div class="m-sub">{s}</div></div>' for l, v, s in items
    )
    html(f'<div class="metric-row">{cards}</div>')


# --------------------------------------------------------------------------
# Shared UI pieces
# --------------------------------------------------------------------------
def sidebar_inputs(algo: str):
    """Draw all inputs in the sidebar. Returns (cfg, train_clicked)."""
    p = algo[:3]  # key prefix so widgets of the two pages never clash

    def side_html(markup: str):
        with st.sidebar:
            html(markup)

    side_html(f'<div class="side-logo">{logo_img(44)}<span>Neural Networks<br>Task1</span></div>')

    side_html('<div class="side-section">Data</div>')
    f1 = st.sidebar.selectbox("Feature 1", FEATURES, index=0, key=f"{p}_f1")
    f2 = st.sidebar.selectbox("Feature 2", FEATURES, index=1, key=f"{p}_f2")
    c1 = st.sidebar.selectbox("Class 1", CLASSES, index=0, key=f"{p}_c1")
    c2 = st.sidebar.selectbox("Class 2", CLASSES, index=1, key=f"{p}_c2")

    side_html('<div class="side-section">Training</div>')
    eta = st.sidebar.number_input("Learning rate (eta)", min_value=0.0, value=0.01,
                                  step=0.001, format="%.4f", key=f"{p}_eta")
    epochs = st.sidebar.number_input("Number of epochs (m)", min_value=1, value=100,
                                     step=1, key=f"{p}_epochs")

    mse_threshold = None
    if algo == "adaline":  # the perceptron has no cost function, so no MSE threshold
        mse_threshold = st.sidebar.number_input("MSE threshold", min_value=0.0, value=0.05,
                                                step=0.01, format="%.4f", key=f"{p}_mse")

    use_bias = st.sidebar.checkbox("Add bias", value=True, key=f"{p}_bias")

    side_html('<div class="side-section">Algorithm</div>')
    st.sidebar.radio("Algorithm", list(ALGO_TITLES.values()),
                     index=0 if algo == "perceptron" else 1, disabled=True,
                     label_visibility="collapsed",
                     help="Chosen on the landing page. Use Back to change it.",
                     key=f"{p}_algo_radio")

    st.sidebar.write("")
    clicked = st.sidebar.button("Train & Test", type="primary", use_container_width=True,
                                key=f"{p}_train")

    cfg = {
        "feature_1": f1, "feature_2": f2, "class_1": c1, "class_2": c2,
        "eta": float(eta), "epochs": int(epochs), "mse_threshold": mse_threshold,
        "use_bias": use_bias,
    }
    return cfg, clicked


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFC2D0")
    ax.tick_params(colors="#555870", labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_boundary(result: dict, cfg: dict) -> Figure:
    fig = Figure(figsize=(6, 5.6))
    ax = fig.subplots()
    _style_axes(ax)

    xtr, ytr = result["x_train"], result["y_train"]
    xte, yte = result["x_test"], result["y_test"]
    all_x = np.vstack([xtr, xte])
    x_min, x_max = all_x[:, 0].min(), all_x[:, 0].max()
    y_min, y_max = all_x[:, 1].min(), all_x[:, 1].max()
    pad_x, pad_y = 0.08 * (x_max - x_min), 0.08 * (y_max - y_min)
    x_lo, x_hi, y_lo, y_hi = x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y

    # shaded decision regions + the line  w1*x1 + w2*x2 + b = 0
    w1, w2, b = result["w1"], result["w2"], result["b"]
    if abs(w1) > 1e-12 or abs(w2) > 1e-12:
        gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, 250), np.linspace(y_lo, y_hi, 250))
        z = w1 * gx + w2 * gy + b
        if z.min() < 0 < z.max():
            ax.contourf(gx, gy, z, levels=[z.min(), 0, z.max()], colors=CLASS_LIGHT, alpha=0.55)
            ax.contour(gx, gy, z, levels=[0], colors=INK, linewidths=2)

    for label, name, color in [(-1, cfg["class_1"], CLASS_COLORS[0]), (1, cfg["class_2"], CLASS_COLORS[1])]:
        m = ytr == label
        ax.scatter(xtr[m, 0], xtr[m, 1], c=color, s=34, edgecolors="white", linewidths=0.6,
                   label=f"{name} (train)", zorder=3)
        m = yte == label
        ax.scatter(xte[m, 0], xte[m, 1], c=color, s=46, marker="X", edgecolors="white", linewidths=0.6,
                   label=f"{name} (test)", zorder=3)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK, lw=2))
    labels = [h.get_label() for h in handles[:-1]] + ["Decision boundary"]
    ax.legend(handles, labels, fontsize=8, framealpha=0.92, loc="best")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(cfg["feature_1"], fontsize=10)
    ax.set_ylabel(cfg["feature_2"], fontsize=10)
    fig.tight_layout()
    return fig


def draw_confusion(result: dict, cfg: dict, color: str) -> Figure:
    conf = np.asarray(result["confusion"])
    fig = Figure(figsize=(6, 5.6))
    ax = fig.subplots()
    cmap = LinearSegmentedColormap.from_list("acc", ["#FFFFFF", color])
    ax.imshow(conf, cmap=cmap, vmin=0, vmax=max(conf.max(), 1), aspect="auto")
    total = max(conf.sum(), 1)
    for i in range(2):
        for j in range(2):
            v = conf[i, j]
            ax.text(j, i, f"{v}\n{v / total * 100:.1f}%", ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white" if v > conf.max() * 0.55 else INK)
    names = [cfg["class_1"], cfg["class_2"]]
    ax.set_xticks([0, 1], labels=[f"Pred: {n}" for n in names])
    ax.set_yticks([0, 1], labels=[f"True: {n}" for n in names])
    ax.tick_params(length=0, labelsize=10, colors="#555870")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return fig


def render_results(algo: str, result: dict, cfg: dict):
    accent = ACCENTS[algo][0]

    top_left, top_right = st.columns([1, 2], gap="medium")
    with top_left:
        metric_cards([
            ("Test accuracy", f"{result['accuracy'] * 100:.2f}%", f"{cfg['class_1']} vs {cfg['class_2']}"),
        ])
    with top_right, st.container(border=True):
        html('<div class="panel-title">Classify a single sample</div>')
        st.caption("Enter the two feature values (in original units) and press Classify.")
        s1, s2 = st.columns(2)
        v1 = s1.number_input(cfg["feature_1"], value=0.0, key=f"{algo}_s1")
        v2 = s2.number_input(cfg["feature_2"], value=0.0, key=f"{algo}_s2")
        if st.button("Classify", type="primary", key=f"{algo}_classify"):
            name = classify_sample(result, cfg, v1, v2)
            color = CLASS_COLORS[0] if name == cfg["class_1"] else CLASS_COLORS[1]
            html(f'<div class="pred-pill" style="background:{color};">Predicted class: <b>{name}</b></div>')

    left, right = st.columns(2)
    with left, st.container(border=True):
        html('<div class="panel-title">Decision boundary</div>')
        st.pyplot(draw_boundary(result, cfg))
    with right, st.container(border=True):
        html('<div class="panel-title">Confusion matrix (test set)</div>')
        st.pyplot(draw_confusion(result, cfg, accent))


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def algo_card(col, algo: str, tag: str, title: str, text: str, chips, key: str, label: str):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    with col:
        html(f"""
        <div class="card algo-card {algo}">
          <div class="card-strip"></div>
          <span class="tag">{tag}</span>
          <div class="card-title">{title}</div>
          <div class="card-text">{text}</div>
          <div class="chips">{chips_html}</div>
        </div>
        """)
        st.button(label, on_click=go, args=(algo,), use_container_width=True, key=key)


def landing_page():
    inject_css(LANDING_ACCENT)
    hero("Single Layer Perceptron &amp; Adaline on the Penguins dataset")

    html('<div class="section-title">Choose your learning algorithm</div>')
    col1, col2 = st.columns(2, gap="large")
    algo_card(col1, "perceptron", "SLP", "Single Layer Perceptron",
              "Hard-limiting neuron. Updates its weights only when a sample is misclassified.",
              ["Signum activation", "Error-driven updates", "No cost function"],
              "open_perc", "Open Perceptron  →")
    algo_card(col2, "adaline", "ADALINE", "Adaline",
              "Adaptive linear neuron. Learns from the continuous error and minimises the MSE.",
              ["Linear activation", "Mean squared error", "Gradient descent"],
              "open_ada", "Open Adaline  →")

    html('<div class="section-title" style="margin-top:18px;">How it works</div>')
    html("""
    <div class="steps">
      <div class="step"><div><span class="step-num">1</span><span class="step-title">Pick the data</span></div>
        <div class="step-text">Choose two features and two penguin species to classify.</div></div>
      <div class="step"><div><span class="step-num">2</span><span class="step-title">Set the training</span></div>
        <div class="step-text">Learning rate, epochs, bias and (for Adaline) the MSE threshold.</div></div>
      <div class="step"><div><span class="step-num">3</span><span class="step-title">Train &amp; evaluate</span></div>
        <div class="step-text">See the decision boundary, confusion matrix and accuracy on 20 test samples per class.</div></div>
    </div>
    """)
    html('<div class="footer-note">Neural Networks &middot; Task 1</div>')


def algorithm_page(algo: str):
    inject_css(ACCENTS[algo])
    cfg, train_clicked = sidebar_inputs(algo)

    top_l, _ = st.columns([1, 3])
    top_l.button("← Back to algorithms", on_click=go, args=(None,), key=f"{algo}_back")
    hero(small=True, title=ALGO_TITLES[algo])

    if train_clicked:
        if cfg["feature_1"] == cfg["feature_2"]:
            st.error("Please select two different features.")
        elif cfg["class_1"] == cfg["class_2"]:
            st.error("Please select two different classes.")
        else:
            with st.spinner("Training..."):
                st.session_state["result"] = train_and_evaluate(algo, cfg)
                st.session_state["cfg"] = cfg  # remember the settings used for this result

    if "result" in st.session_state:
        render_results(algo, st.session_state["result"], st.session_state["cfg"])
    else:
        st.info("Choose your settings in the sidebar, then press **Train & Test**.")
    html('<div class="footer-note">Neural Networks &middot; Task 1</div>')


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def _page_icon():
    try:
        from PIL import Image
        return Image.open(LOGO_PATH)
    except Exception:
        return "🧠"


st.set_page_config(page_title=APP_TITLE, page_icon=_page_icon(), layout="wide")
st.session_state.setdefault("page", None)

page = st.session_state["page"]
if page in ALGO_TITLES:
    algorithm_page(page)
else:
    landing_page()