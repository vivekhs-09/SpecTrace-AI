"""
AI-Based Product Identification from Engineering Drawings
Streamlit UI — drag and drop, all 3 AWS services, efficiency scores, final answer
"""

import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from product_extractor import ProductExtractor, ServiceResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Product Identifier — Engineering Drawings",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title    { font-size:26px; font-weight:700; color:#1a1a2e; margin-bottom:2px; }
    .sub-title     { font-size:13px; color:#666; margin-bottom:12px; }
    .section-hdr   { font-size:15px; font-weight:600; color:#1a1a2e; margin:14px 0 6px; }
    .score-card    { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                     padding:16px; text-align:center; }
    .score-num     { font-size:42px; font-weight:700; }
    .score-label   { font-size:12px; color:#666; margin-top:2px; }
    .winner-badge  { background:#16a34a; color:#fff; border-radius:4px;
                     padding:2px 8px; font-size:11px; font-weight:600; }
    .final-box     { background:#f0fdf4; border:1px solid #86efac;
                     border-radius:8px; padding:20px; }
    .product-row   { background:#fff; border:1px solid #e2e8f0; border-radius:6px;
                     padding:12px 16px; margin-bottom:8px; }
    div[data-testid="stMetricValue"] { font-size:18px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Efficiency score computation
# ---------------------------------------------------------------------------

def compute_efficiency(result: ServiceResult) -> dict:
    """
    Returns a 0-100 efficiency score per service, broken into four components:

    Extraction (30%)     — how completely text was captured
    Identification (40%) — how directly products were identified (0 if needs manual work)
    Confidence (20%)     — average confidence of results
    Directness (10%)     — no post-processing needed = 100, post-processing needed = 0
    """
    if result.error:
        return dict(overall=0, extraction=0, identification=0, confidence=0, directness=0)

    if result.query_answers:                          # Textract Queries
        found   = sum(1 for qa in result.query_answers if qa.answer != "NOT FOUND")
        total   = len(result.query_answers)
        extraction     = round((found / total) * 100, 1)
        identification = extraction                   # each attribute = direct product info
        confidence     = round(
            sum(qa.confidence for qa in result.query_answers if qa.answer != "NOT FOUND")
            / max(found, 1), 1
        )
        directness = 100

    elif result.products:                             # Bedrock (has products)
        extraction     = min(100.0, round(len(result.text_lines) / 15 * 100, 1))
        identification = min(100.0, round(len(result.products) * 34, 1))
        confidence     = round(
            sum(p.confidence for p in result.products) / len(result.products) * 100, 1
        )
        directness = 100

    else:                                             # Rekognition (raw text only)
        extraction     = min(100.0, round(len(result.text_lines) / 15 * 100, 1))
        identification = 0      # cannot identify products without an extra step
        confidence     = round(
            sum(t.confidence for t in result.text_lines) / max(len(result.text_lines), 1), 1
        )
        directness = 0          # needs manual filtering

    overall = round(
        extraction     * 0.30 +
        identification * 0.40 +
        confidence     * 0.20 +
        directness     * 0.10,
        1,
    )
    return dict(
        overall=overall,
        extraction=extraction,
        identification=identification,
        confidence=confidence,
        directness=directness,
    )


def score_color(score: float) -> str:
    if score >= 75:
        return "#16a34a"
    if score >= 45:
        return "#d97706"
    return "#dc2626"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### AWS Configuration")

    region = st.selectbox(
        "AWS Region",
        ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        index=0,
    )
    bedrock_model = st.selectbox(
        "Bedrock Model",
        [
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        ],
    )

    st.divider()
    st.markdown("**Required IAM Permissions**")
    st.code(
        "AmazonTextractFullAccess\n"
        "AmazonRekognitionFullAccess\n"
        "AmazonBedrockFullAccess",
        language=None,
    )
    st.caption("Bedrock: AWS Console → Bedrock → Model Access → Enable Claude")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">AI-Based Product Identification from Engineering Drawings</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a drawing — all three AWS services run automatically and are scored for efficiency</div>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Evaluation criteria (collapsed by default)
# ---------------------------------------------------------------------------
with st.expander("Evaluation Criteria — How the three services are compared"):
    crit = {
        "Criteria": [
            "Mechanism",
            "Product identification",
            "Requires post-processing",
            "Attribute targeting",
            "Handles complex layouts",
            "Approx. cost per image",
        ],
        "AWS Textract Queries": [
            "OCR + targeted Q&A in one API call",
            "Direct — returns answers per attribute",
            "No",
            "Yes — attributes defined upfront",
            "Yes — understands document structure",
            "~ $0.015",
        ],
        "AWS Rekognition": [
            "Visual scene-text detection",
            "No — all text returned equally",
            "Yes — manual filtering required",
            "No",
            "Limited for dense documents",
            "~ $0.001",
        ],
        "AWS Bedrock Claude": [
            "Multimodal LLM — full image understanding",
            "Direct — structured product list",
            "No",
            "Yes — via prompt",
            "Yes — best engineering context",
            "~ $0.04",
        ],
    }
    st.dataframe(pd.DataFrame(crit).set_index("Criteria"), use_container_width=True, height=260)

st.divider()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown('<div class="section-hdr">Upload Engineering Drawing</div>', unsafe_allow_html=True)

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

uploaded = st.file_uploader(
    "Drag and drop or click to browse — JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
    key=f"uploader_{st.session_state['upload_key']}",
)

if not uploaded:
    st.info("Upload an engineering drawing to begin analysis.")
    st.stop()

image_bytes = uploaded.read()
image = Image.open(io.BytesIO(image_bytes))

col_img, col_meta = st.columns([2, 1])
with col_img:
    st.image(image, caption=uploaded.name, use_container_width=True)
with col_meta:
    st.markdown("**Image Details**")
    st.markdown(f"- **File:** {uploaded.name}")
    st.markdown(f"- **Size:** {len(image_bytes)//1024} KB")
    st.markdown(f"- **Dimensions:** {image.width} x {image.height} px")
    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    if st.button("Upload New Image", use_container_width=True):
        st.session_state["upload_key"] += 1
        st.session_state.pop("results", None)
        st.rerun()

if not run_btn and "results" not in st.session_state:
    st.stop()

# ---------------------------------------------------------------------------
# Run all three services
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state.pop("results", None)

    extractor = ProductExtractor(region=region, bedrock_model=bedrock_model)
    results   = {}
    progress  = st.progress(0, text="Starting ...")

    with st.spinner("Running AWS Textract Queries ..."):
        progress.progress(10, text="[1 of 3] AWS Textract Queries")
        results["textract"] = extractor._run_textract(image_bytes)
        progress.progress(33)

    with st.spinner("Running AWS Rekognition ..."):
        progress.progress(40, text="[2 of 3] AWS Rekognition DetectText")
        results["rekognition"] = extractor._run_rekognition(image_bytes)
        progress.progress(66)

    with st.spinner("Running AWS Bedrock Claude ..."):
        mt = "image/png" if uploaded.name.lower().endswith(".png") else "image/jpeg"
        progress.progress(70, text="[3 of 3] AWS Bedrock Claude Vision")
        results["bedrock"] = extractor._run_bedrock(image_bytes, mt)
        progress.progress(100, text="Complete")

    st.session_state["results"] = results
    progress.empty()

results: dict[str, ServiceResult] = st.session_state["results"]

# ---------------------------------------------------------------------------
# Compute scores
# ---------------------------------------------------------------------------
scores = {k: compute_efficiency(v) for k, v in results.items()}
best   = max(scores, key=lambda k: scores[k]["overall"])

# ---------------------------------------------------------------------------
# Efficiency Score Summary — top of results
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-hdr">Efficiency Scores</div>', unsafe_allow_html=True)
st.caption("Overall score = Extraction (30%) + Product Identification (40%) + Confidence (20%) + Directness (10%)")

labels = {
    "textract":    "AWS Textract Queries",
    "rekognition": "AWS Rekognition",
    "bedrock":     "AWS Bedrock Claude",
}

col1, col2, col3 = st.columns(3)
for col, (key, label) in zip([col1, col2, col3], labels.items()):
    sc  = scores[key]
    col.markdown(
        f"""
        <div class="score-card">
            <div style="font-size:13px;font-weight:600;margin-bottom:6px;">{label}</div>
            <div class="score-num" style="color:{score_color(sc['overall'])};">{sc['overall']}</div>
            <div class="score-label">/ 100 overall</div>
            {"<br><span class='winner-badge'>Best Performer</span>" if key == best else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Score breakdown bar chart
fig_scores = go.Figure()
components  = ["extraction", "identification", "confidence", "directness"]
comp_labels = ["Extraction (30%)", "Identification (40%)", "Confidence (20%)", "Directness (10%)"]
colors_svc  = {"textract": "#2563eb", "rekognition": "#d97706", "bedrock": "#16a34a"}

for key, label in labels.items():
    fig_scores.add_trace(go.Bar(
        name=label,
        x=comp_labels,
        y=[scores[key][c] for c in components],
        marker_color=colors_svc[key],
    ))

fig_scores.update_layout(
    barmode="group",
    height=300,
    margin=dict(t=10, b=10),
    yaxis=dict(title="Score (0-100)", range=[0, 110]),
    xaxis_title="",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=-0.25),
)
st.plotly_chart(fig_scores, use_container_width=True)

# ---------------------------------------------------------------------------
# Detailed results tabs
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-hdr">Detailed Results</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Textract Queries",
    "Rekognition DetectText",
    "Bedrock Claude",
])

# ── Textract ──────────────────────────────────────────────────────────────
with tab1:
    r  = results["textract"]
    sc = scores["textract"]
    if r.error:
        st.error(f"Error: {r.error}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Score",     f"{sc['overall']} / 100")
        m2.metric("Attributes Found",  f"{sum(1 for qa in r.query_answers if qa.answer != 'NOT FOUND')} / {len(r.query_answers)}")
        m3.metric("Avg Confidence",    f"{sc['confidence']}%")
        m4.metric("Latency",           f"{r.latency_ms} ms")

        st.markdown("**Attribute extraction results:**")
        rows = []
        for qa in r.query_answers:
            found = qa.answer != "NOT FOUND"
            rows.append({
                "Attribute":          qa.alias,
                "Question":           qa.question,
                "Extracted Value":    qa.answer,
                "Confidence":         f"{qa.confidence}%" if found else "—",
                "Status":             "Found" if found else "Not Found",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Attribute"), use_container_width=True)

# ── Rekognition ────────────────────────────────────────────────────────────
with tab2:
    r  = results["rekognition"]
    sc = scores["rekognition"]
    if r.error:
        st.error(f"Error: {r.error}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Score",     f"{sc['overall']} / 100")
        m2.metric("Text Lines Found",  len(r.text_lines))
        m3.metric("Avg Confidence",    f"{sc['confidence']}%")
        m4.metric("Latency",           f"{r.latency_ms} ms")

        st.warning(
            "Rekognition returns all text without classification. "
            "A separate filtering step is needed to identify products — "
            "this is reflected in the lower efficiency score."
        )
        if r.text_lines:
            st.dataframe(
                pd.DataFrame([
                    {"Detected Text": t.text, "Confidence": f"{t.confidence}%"}
                    for t in r.text_lines
                ]),
                use_container_width=True,
                height=320,
            )

# ── Bedrock ────────────────────────────────────────────────────────────────
with tab3:
    r  = results["bedrock"]
    sc = scores["bedrock"]
    if r.error:
        st.error(f"Error: {r.error}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Score",      f"{sc['overall']} / 100")
        m2.metric("Products Found",     len(r.products))
        m3.metric("Avg Confidence",     f"{sc['confidence']}%")
        m4.metric("Latency",            f"{r.latency_ms} ms")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**All text extracted:**")
            for t in r.text_lines:
                st.markdown(f"- {t.text}")
        with col_b:
            st.markdown("**Products identified:**")
            for p in r.products:
                with st.container(border=True):
                    st.markdown(f"**{p.name}** — `{p.model_number}`")
                    st.caption(f"Confidence: {p.confidence:.0%}   |   {p.context}")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Final Answer
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-hdr">Final Answer</div>', unsafe_allow_html=True)

bedrock_r  = results["bedrock"]
textract_r = results["textract"]
rekog_r    = results["rekognition"]

# ── 1. All Text Extracted ──────────────────────────────────────────────────
st.markdown("### 1. All Text Extracted from Image")

# Collect all unique text across all three services
all_text = set()

# From Textract — answered attributes
for qa in textract_r.query_answers:
    if qa.answer != "NOT FOUND":
        all_text.add(qa.answer)

# From Rekognition — raw detected lines
for t in rekog_r.text_lines:
    all_text.add(t.text)

# From Bedrock — extracted text list
for t in bedrock_r.text_lines:
    all_text.add(t.text)

if all_text:
    text_df = pd.DataFrame(sorted(all_text), columns=["Extracted Text"])
    st.dataframe(text_df, use_container_width=True, height=250)
else:
    st.info("No text extracted from the image.")

# ── 2. Product Name / Model Number Identified ─────────────────────────────
st.markdown("### 2. Product Name / Model Number Identified")

if bedrock_r.error:
    st.error(f"Bedrock error: {bedrock_r.error}")
elif bedrock_r.products:
    # Pick the single most specific product — longest model number wins
    best_product = max(bedrock_r.products, key=lambda p: len(p.model_number))

    st.markdown(
        f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                    padding:20px;margin-bottom:12px;">
            <div style="font-size:26px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">
                {best_product.name}
            </div>
            <div style="font-size:14px;color:#374151;">
                Model Number : <strong>{best_product.model_number}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cross-service confirmation
    st.markdown("**Confirmed across services:**")
    textract_answers = [qa.answer for qa in textract_r.query_answers if qa.answer != "NOT FOUND"]
    rekog_lines      = [t.text for t in rekog_r.text_lines]

    in_textract = any(best_product.model_number.lower() in t.lower() for t in textract_answers)
    in_rekog    = any(best_product.model_number.lower() in t.lower() for t in rekog_lines)

    val_rows = [{
        "Product / Model Number": f"{best_product.name} — {best_product.model_number}",
        "Textract":               "Found" if in_textract else "Not found",
        "Rekognition":            "Found" if in_rekog    else "Not found",
        "Bedrock Claude":         "Found",
    }]

    st.dataframe(
        pd.DataFrame(val_rows).set_index("Product / Model Number"),
        use_container_width=True,
    )
else:
    st.warning("No products identified. Try uploading a clearer image.")

# ── Recommendation ─────────────────────────────────────────────────────────
winner_label = labels[best]
winner_score = scores[best]["overall"]
st.divider()
st.success(
    f"**Best performing service:** {winner_label} "
    f"(efficiency score: {winner_score} / 100). "
    + (
        "Textract Queries is fastest and lowest cost when target attributes are known in advance."
        if best == "textract" else
        "Bedrock Claude delivered the highest accuracy with full contextual understanding."
        if best == "bedrock" else
        "Rekognition provided the best text coverage — pair it with Bedrock for product classification."
    )
)
