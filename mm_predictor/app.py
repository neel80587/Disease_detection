import streamlit as st
from PIL import Image
import predictors as P

st.set_page_config(page_title="Multimodal Disease Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Multimodal Human Disease Predictor")
st.warning("Research demo, not medical advice. Models are small, trained on limited public data (one on synthetic data) and not clinically validated.")
tab1, tab2, tab3 = st.tabs(["Heart disease", "Skin lesion", "Common symptoms (demo)"])

def num(label, lo, hi, val, step=1.0): return st.number_input(label, lo, hi, val, step)

with tab1:
    st.caption("UCI Heart Disease data (920 patients, 4 hospitals). Untick a group if you don't have those results.")
    with st.expander("Clinical", expanded=True):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 100, 55); sex = c2.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female")
        cp = c1.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda v: ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][v - 1])
        bp = c2.number_input("Resting BP (mmHg)", 70, 250, 130); chol = c1.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = c2.selectbox("Fasting sugar > 120", [0, 1], format_func=lambda v: "Yes" if v else "No")
    use_ecg = st.checkbox("I have exercise / ECG results")
    ecg = {}
    if use_ecg:
        c1, c2 = st.columns(2)
        ecg = dict(restecg=c1.selectbox("Resting ECG", [0, 1, 2], format_func=lambda v: ["Normal", "ST-T abnormality", "LV hypertrophy"][v]),
                   thalach=c2.number_input("Max heart rate", 60, 220, 150), exang=c1.selectbox("Exercise angina", [0, 1], format_func=lambda v: "Yes" if v else "No"),
                   oldpeak=c2.number_input("ST depression", 0.0, 7.0, 1.0, 0.1), slope=c1.selectbox("ST slope", [1, 2, 3], format_func=lambda v: ["Upsloping", "Flat", "Downsloping"][v - 1]))
    use_img = st.checkbox("I have scan results")
    scan = {}
    if use_img:
        c1, c2 = st.columns(2)
        scan = dict(ca=c1.selectbox("Vessels coloured by fluoroscopy", [0, 1, 2, 3]), thal=c2.selectbox("Thallium scan", [3, 6, 7], format_func=lambda v: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[v]))
    if st.button("Predict heart risk"):
        use = ["clinical"] + (["exercise_ecg"] if use_ecg else []) + (["imaging_scan"] if use_img else [])
        r = P.predict_heart(age, sex, cp, bp, chol, fbs, use=use, **ecg, **scan)
        st.metric("Probability of heart disease", f"{r['probability']:.0%}"); st.bar_chart(r["per_modality"])

with tab2:
    info = P.skin_info()
    st.caption("Dermoscopy images only (close-up, magnified clinical images). Ordinary phone photos give meaningless scores.")
    age_v = sex_v = site_v = None
    if info["fine_tuned"]:
        c1, c2, c3 = st.columns(3)
        if c1.checkbox("Add age"): age_v = c1.number_input("Age", 1, 100, 50)
        sex_v = c2.selectbox("Sex", ["unknown", "male", "female"]); site_v = c3.selectbox("Body site", P.SKIN_SITES, index=P.SKIN_SITES.index("unknown"))
    f = st.file_uploader("Lesion image", type=["jpg", "jpeg", "png"])
    if f:
        im = Image.open(f); st.image(im, width=280)
        try:
            r = P.predict_skin(im, age_v, sex_v, site_v); st.metric("Estimated probability malignant", f"{r['probability']:.0%}")
            if r["model"] == "fine-tuned":
                (st.error if r["flagged"] else st.success)("Above the follow-up threshold (tuned to catch about 90% of malignant lesions on validation data)." if r["flagged"] else "Below the follow-up threshold.")
                if not r["uses_metadata"]: st.caption("This model does not use age, sex or site.")
            else:
                st.caption(f"Legacy model: deep features {r['deep']:.0%} · handcrafted {r['handcrafted']:.0%}")
        except ImportError as e:
            st.error(f"Skin model needs extra packages: {e}. Run: python -m pip install torch timm scikit-image scipy")

with tab3:
    st.caption("Trained on synthetic data generated from a hand-written prior, so this is a demo of the pipeline only.")
    sel = st.multiselect("Symptoms", list(P.SYMPTOM_LABELS), format_func=P.SYMPTOM_LABELS.get)
    note = st.text_area("Notes", placeholder="e.g. chills and body aches since yesterday")
    c = st.columns(5); v = [x.number_input(n, 0.0, 400.0, 0.0) for x, n in zip(c, ["Temp °C", "HR", "SpO₂", "Sys BP", "Glucose"])]
    if st.button("Predict condition"):
        v = [x or None for x in v]
        for name, p in P.predict_symptoms(sel, note, *v): st.progress(min(p, 1.0), text=f"{name}: {p:.0%}")
