import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="DigitDraw",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "digit_cnn.keras"
ACCENT = "#818cf8"

for key, value in {
    "canvas_key": 0,
    "upload_key": 0,
    "history": [],
    "show_fix": False,
}.items():
    st.session_state.setdefault(key, value)


# ---------------- STYLING ----------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp { background: #0a0a0b; color: #e5e5e5; }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer, [data-testid="stToolbar"] { display: none; }

        .block-container { padding-top: 2.5rem; max-width: 1100px; }

        section[data-testid="stSidebar"] {
            background: #0f0f11;
            border-right: 1px solid #1f1f23;
        }

        .title { font-size: 30px; font-weight: 700; color: #fafafa; letter-spacing: -0.02em; }
        .sub { color: #71717a; font-size: 14px; margin: 2px 0 26px 0; }

        .card {
            background: #111113;
            border: 1px solid #232326;
            border-radius: 16px;
            padding: 22px 24px;
        }
        .label {
            color: #71717a; font-size: 12px; font-weight: 500;
            letter-spacing: 0.08em; text-transform: uppercase;
        }
        .big-digit {
            font-size: 84px; font-weight: 700; line-height: 1.05;
            color: #fafafa; margin: 6px 0 2px 0;
        }
        .conf { color: #a1a1aa; font-size: 14px; }
        .conf b { color: #818cf8; font-weight: 600; }

        h4 { color: #fafafa; font-weight: 600; letter-spacing: -0.01em; }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid #2a2a2f;
            background: #16161a;
            color: #e5e5e5;
            font-weight: 500;
            min-height: 42px;
        }
        .stButton > button:hover { border-color: #818cf8; color: #ffffff; }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: #818cf8; border-color: #818cf8; color: #0a0a0b;
        }

        [data-testid="stFileUploader"] section {
            background: #111113; border: 1px dashed #2f2f35; border-radius: 14px;
        }
        iframe { border-radius: 14px; }
        [data-testid="stMetricValue"] { color: #fafafa; }
        [data-testid="stMetricLabel"] { color: #71717a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- MODEL ----------------

@st.cache_resource
def get_model(path):
    return load_model(path)


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}\n\nPlace digit_cnn.keras next to app.py.")
    st.stop()

try:
    model = get_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ---------------- HELPERS ----------------

def preprocess_digit(gray):
    """
    White digit on black (2D uint8) -> MNIST-style 28x28 float array in 0..1.
    Tight crop, scale longest side to 20 px, center by center of mass.
    """
    ys, xs = np.where(gray > 30)
    if xs.size == 0:
        return None

    digit = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = digit.shape

    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit_img = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas28 = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas28[oy:oy + new_h, ox:ox + new_w] = np.asarray(digit_img, dtype=np.float32)

    total = canvas28.sum()
    if total > 0:
        yy, xx = np.indices(canvas28.shape)
        cy = (yy * canvas28).sum() / total
        cx = (xx * canvas28).sum() / total
        canvas28 = np.roll(
            canvas28, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1)
        )

    return np.clip(canvas28 / 255.0, 0.0, 1.0)


def preprocess_upload(file):
    """Uploaded photo/image -> 28x28 array. Auto-inverts dark-on-light digits."""
    img = Image.open(file)
    img.thumbnail((600, 600))

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.split()[-1])
        img = background

    gray = np.asarray(img.convert("L"), dtype=np.float32)

    # If the border is light, the digit is dark on light paper: invert.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 127:
        gray = 255.0 - gray

    lo, hi = gray.min(), gray.max()
    if hi - lo < 30:
        return None

    gray = (gray - lo) / (hi - lo) * 255.0
    gray[gray < 60] = 0  # suppress faint background noise
    return preprocess_digit(gray.astype(np.uint8))


def to_probabilities(output):
    output = np.asarray(output, dtype=np.float64)
    if np.all(output >= 0) and abs(output.sum() - 1.0) < 1e-3:
        return output
    exp = np.exp(output - output.max())
    return exp / exp.sum()


def predict(image_array):
    x = image_array.reshape(1, 28, 28, 1).astype(np.float32)
    return to_probabilities(model(x, training=False).numpy()[0])


# ---------------- CALLBACKS ----------------

def clear_draw():
    st.session_state.canvas_key += 1
    st.session_state.show_fix = False


def clear_upload():
    st.session_state.upload_key += 1
    st.session_state.show_fix = False


def ask_fix():
    st.session_state.show_fix = True


def submit_feedback(pred, actual, confidence, source):
    if actual is None:
        actual = st.session_state.get("actual_digit", 0)
    st.session_state.history.append(
        {
            "Source": source,
            "Predicted": int(pred),
            "Actual": int(actual),
            "Confidence": round(confidence, 1),
            "Correct": int(pred) == int(actual),
        }
    )
    st.session_state.show_fix = False
    if source == "Draw":
        st.session_state.canvas_key += 1
    else:
        st.session_state.upload_key += 1
    st.toast("Feedback saved")


def reset_stats():
    st.session_state.history = []


# ---------------- SIDEBAR ----------------

history = st.session_state.history
total = len(history)
correct = sum(1 for row in history if row["Correct"])

with st.sidebar:
    st.markdown("### 🔢 DigitDraw")
    st.caption("Handwritten digit recognition with a CNN.")
    st.divider()

    st.markdown('<div class="label">Session accuracy</div>', unsafe_allow_html=True)
    st.metric(
        "Accuracy",
        f"{correct / total * 100:.0f}%" if total else "—",
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Checked", total)
    col_b.metric("Correct", correct)

    if total:
        with st.expander("History"):
            table = pd.DataFrame(history[::-1][:15])
            table["Correct"] = table["Correct"].map({True: "✓", False: "✗"})
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.button("Reset stats", on_click=reset_stats, use_container_width=True)
    else:
        st.caption("Mark predictions as right or wrong to track how well the model does.")

    st.divider()
    st.caption("Input 28 × 28 · Classes 0–9")


# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">DigitDraw</div>'
    '<div class="sub">Draw a digit and watch the model predict it live.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")


# ---------------- INPUT ----------------

image_array = None

with left:
    mode = st.radio(
        "Input",
        ["Draw", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Draw":
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"digit_canvas_{st.session_state.canvas_key}",
            update_streamlit=True,
            return_image_data=True,
        )
        st.caption("Draw one big, bold digit. The prediction updates after every stroke.")
        st.button("Clear canvas", on_click=clear_draw, use_container_width=True)

        if canvas_result.image_data is not None:
            rgba = np.asarray(canvas_result.image_data).clip(0, 255).astype(np.uint8)
            gray = np.asarray(Image.fromarray(rgba).convert("L"), dtype=np.uint8)
            image_array = preprocess_digit(gray)

    else:
        uploaded = st.file_uploader(
            "Upload a digit image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key=f"upload_{st.session_state.upload_key}",
            label_visibility="collapsed",
        )
        st.caption("Best with a single digit on a plain background, light or dark.")

        if uploaded is not None:
            try:
                st.image(uploaded, width=220)
                image_array = preprocess_upload(uploaded)
                if image_array is None:
                    st.warning("Couldn't find a clear digit in that image.")
            except Exception as error:
                st.error(f"Could not read that image: {error}")
            st.button("Remove image", on_click=clear_upload, use_container_width=True)


# ---------------- RESULT ----------------

with right:
    if image_array is None:
        st.markdown(
            """
            <div class="card">
                <div class="label">Prediction</div>
                <div class="big-digit" style="color:#3f3f46;">–</div>
                <div class="conf">Waiting for a digit…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        probs = predict(image_array)
        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

        card_col, img_col = st.columns([1.4, 1], gap="medium")
        with card_col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="label">Prediction</div>
                    <div class="big-digit">{pred}</div>
                    <div class="conf">Confidence <b>{conf:.1f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with img_col:
            preview = Image.fromarray((image_array * 255).astype(np.uint8)).resize(
                (112, 112), Image.Resampling.NEAREST
            )
            st.image(preview, caption="Model input (28×28)", width=112)

        if conf < 60:
            st.caption("Low confidence. Try a larger, clearer, bolder digit.")

        colors = [ACCENT if i == pred else "#2a2a2f" for i in range(10)]
        fig = go.Figure(
            go.Bar(
                x=[str(i) for i in range(10)],
                y=probs * 100,
                marker_color=colors,
                text=[f"{p * 100:.0f}%" if p > 0.005 else "" for p in probs],
                textposition="outside",
                textfont=dict(color="#a1a1aa", size=11),
                cliponaxis=False,
                hovertemplate="Digit %{x}: %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=24, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a1a1aa"),
            xaxis=dict(showgrid=False, linecolor="#232326"),
            yaxis=dict(visible=False, range=[0, 115]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Was this right?")
        yes_col, no_col = st.columns(2)
        yes_col.button(
            "✓ Yes",
            type="primary",
            use_container_width=True,
            on_click=submit_feedback,
            args=(pred, pred, conf, mode),
        )
        no_col.button("✗ No", use_container_width=True, on_click=ask_fix)

        if st.session_state.show_fix:
            st.radio(
                "What digit was it?",
                [d for d in range(10) if d != pred],
                horizontal=True,
                key="actual_digit",
            )
            st.button(
                "Save correction",
                use_container_width=True,
                on_click=submit_feedback,
                args=(pred, None, conf, mode),
            )


# ---------------- FOOTER ----------------

st.divider()
st.caption("DigitDraw · Python, Streamlit and TensorFlow")