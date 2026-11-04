import joblib
import streamlit as st

st.set_page_config(page_title="Spam Detector", page_icon="📧")

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

st.title("📧 Spam Detector")
st.write("Paste a message to check whether it looks like spam.")

text = st.text_area("Message", height=150, placeholder="Paste a message...")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        p = float(model.predict_proba([text])[0][1])
        if p >= 0.5:
            st.error(f"SPAM ({p:.1%} spam probability)")
        else:
            st.success(f"Not spam ({p:.1%} spam probability)")
        st.progress(p)
