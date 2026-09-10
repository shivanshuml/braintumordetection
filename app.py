# app.py
import streamlit as st
import pandas as pd
from PIL import Image

# Import from modules
from styles import inject_css
from model import (
    load_tumor_model,
    load_validator_model,
    validate_mri,
    predict,
    generate_gradcam,
    LABELS,
    TUMOR_INFO,
    VALIDATION_DESCRIPTIONS
)
from pdf_report import generate_pdf

# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier & Validator",
    page_icon="🧠",
    layout="wide"
)

inject_css()

# Header Banner
try:
    st.image("assets/banner.png", use_container_width=True)
except Exception:
    pass

# ==========================================
# LOAD MODELS
# ==========================================
try:
    tumor_model = load_tumor_model()
except Exception as e:
    st.error(f"Tumor classification model loading failed: {e}")
    st.stop()

validator_model = load_validator_model()

# ==========================================
# HEADER & INFORMATION
# ==========================================
st.markdown("""
<div style="text-align:center">
<h1 class="main-title">🧠 Brain Tumor MRI Classifier</h1>
<p class="subtitle">Two-Stage AI Pipeline: MRI Verification & EfficientNetB0 Tumor Classification</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload brain MRI scans. The system executes a **Two-Stage Validation Pipeline**:
1. **Stage 1 (MRI Validator):** Confirms if the upload is an authentic **Brain MRI** and rejects non-brain/non-MRI images.
2. **Stage 2 (Tumor Classifier):** Predicts tumor pathology (`Glioma`, `Meningioma`, `No Tumor`, `Pituitary`), generates Grad-CAM explainability heatmaps, and produces diagnostic PDF reports.
""")

st.warning(
    """
    ⚠️ Medical Disclaimer:
    This AI model is intended for educational and research purposes only.
    It is NOT a substitute for professional medical diagnosis.
    Always consult a qualified radiologist or healthcare professional.
    """
)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📋 Patient Information")
patient_name = st.sidebar.text_input("Patient Name (Optional)")
case_id = st.sidebar.text_input("Case ID (Optional)")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Pipeline Status")
st.sidebar.markdown("**Stage 1 Validator:** `Active (Intelligent MRI Verification Engine)`")
st.sidebar.markdown("**Stage 2 Classifier:** `Active (EfficientNetB0 Tumor Diagnostic Model)`")

# ==========================================
# FILE UPLOADER
# ==========================================
uploaded_files = st.file_uploader(
    "Upload MRI Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================================
# TWO-STAGE PROCESSING PIPELINE
# ==========================================
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        # File size check (5MB limit)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error(f"File {uploaded_file.name} is too large. Max allowed size is 5MB.")
            continue

        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load image {uploaded_file.name}: {e}")
            continue

        st.markdown("---")
        st.subheader(f"📁 Scan Analysis: {uploaded_file.name}")

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 📷 Uploaded Scan Preview")
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_info:
            # -------------------------------------------------------------
            # STAGE 1: VALIDATION
            # -------------------------------------------------------------
            st.markdown("#### 🛡️ Stage 1: MRI Input Verification")
            with st.spinner("Validating scan type..."):
                val_result = validate_mri(image, validator_model)

            if not val_result["is_valid"]:
                # --- REJECTION UI ---
                st.markdown(
                    f"""
                    <div class='reject-card'>
                        <h3 style='color: #FF4B4B; margin-top:0;'>❌ Upload Rejected</h3>
                        <p><b>Detected Category:</b> <span style='color:#FFBABA;'>{val_result['predicted_class']}</span> ({val_result['confidence']:.1f}% confidence)</p>
                        <p><b>Reason:</b> {val_result['reason']}</p>
                        <p style='font-size: 13px; color: #d6d6d6; margin-bottom:0;'>
                            <i>Action Required: Please upload a valid Cranial/Brain MRI scan to perform tumor diagnosis.</i>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show validator probability distribution
                val_df = pd.DataFrame({
                    "Category": list(val_result["probabilities"].keys()),
                    "Probability": list(val_result["probabilities"].values())
                })
                st.markdown("**Validator Confidence Breakdown:**")
                st.bar_chart(val_df.set_index("Category"))

                st.info("ℹ️ Stage 2 Tumor Classification skipped because image is not a Brain MRI scan.")
                continue

            else:
                # --- VERIFIED BADGE ---
                st.markdown(
                    f"""
                    <div class='verified-badge'>
                        ✓ Verified Brain MRI ({val_result['confidence']:.1f}% confidence)
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------------------------------------------------------------
            # STAGE 2: TUMOR PREDICTION
            # -------------------------------------------------------------
            st.markdown("#### 🔬 Stage 2: Tumor Classification")
            with st.spinner("Running EfficientNetB0 tumor diagnostic model..."):
                try:
                    predicted_label, confidence, probabilities = predict(tumor_model, image)
                except Exception as e:
                    st.error(f"Prediction error for {uploaded_file.name}: {e}")
                    continue

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if confidence < 60.0:
                st.warning(f"Prediction: {predicted_label} (Low Confidence)")
            else:
                st.success(f"Prediction: {predicted_label}")

            st.metric("Model Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='info-card'>
                <h4 style='margin-top:0;'>📋 Tumor Information</h4>
                <p>{TUMOR_INFO[predicted_label]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------------------
        # GRAD-CAM & PROBABILITY DISTRIBUTION
        # -------------------------------------------------------------
        col_xai, col_chart = st.columns([1, 1])

        with col_xai:
            st.subheader("🔍 Grad-CAM Localization (Explainable AI)")
            with st.spinner("Generating attention heatmap..."):
                gradcam_img = generate_gradcam(tumor_model, image, class_index=LABELS.index(predicted_label))

            if gradcam_img is not None:
                st.image(gradcam_img, caption=f"Visual attention highlighting regions indicative of {predicted_label}", use_container_width=True)
            else:
                st.info("Grad-CAM visualization not available for this layer.")

        with col_chart:
            st.subheader("📊 Tumor Class Probabilities")
            df = pd.DataFrame({
                "Tumor Type": LABELS,
                "Probability": probabilities
            })
            st.bar_chart(df.set_index("Tumor Type"))
            st.table(df.style.format({"Probability": "{:.4f}"}))

        # -------------------------------------------------------------
        # CLINICAL PDF REPORT GENERATION
        # -------------------------------------------------------------
        val_meta = f"Verified Brain MRI ({val_result['confidence']:.1f}%)"
        pdf = generate_pdf(
            prediction=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            labels=LABELS,
            image=image,
            case_id=case_id,
            patient_name=patient_name,
            validation_info=val_meta
        )

        st.download_button(
            label=f"📄 Download Diagnostic Report ({uploaded_file.name})",
            data=pdf,
            file_name=f"Brain_Tumor_Report_{idx+1}.pdf",
            mime="application/pdf",
            key=f"download_{idx}"
        )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class='footer'>
Two-Stage Medical AI Pipeline • TensorFlow • EfficientNetB0 • Streamlit
</div>
""", unsafe_allow_html=True)