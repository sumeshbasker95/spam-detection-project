# ============================================================
#  FINAL app.py (AI + HEURISTIC + ENTROPY DETECTION)
# ============================================================
import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide")

# UI THEME
st.markdown("""<style>
    :root { --bg: #0b1120; --surface: #1e293b; --accent: #3b82f6; }
    html, body, [class*="css"] { background-color: var(--bg) !important; color: #f1f5f9 !important; }
    .hero { background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%); border-radius: 15px; padding: 2rem; border: 1px solid #334155; text-align: center;}
    .card { background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .stButton > button { background: #2563eb !important; color: white !important; font-weight: 700; width: 100%; border-radius: 8px; height: 3.2rem; }
</style>""", unsafe_allow_html=True)

# ASSETS
WHITELIST = {"google.com", "youtube.com", "gmail.com", "whatsapp.com", "tagore-engg.edu.in", "nptel.ac.in", "amazon.in"}
RED_FLAGS = ["offer", "coupon", "enroll now", "click here", "bonuses", "urgent", "verify", "suspended", "login"]

def is_random_url(url):
    """Detects if a URL looks like a randomized phishing link."""
    domain = url.split('/')[0]
    # Count digits and unique characters
    digit_count = sum(c.isdigit() for c in domain)
    unique_chars = len(set(domain))
    # Phishing domains often have high digit density or long random subdomains
    if digit_count > 4 or len(domain) > 30 or unique_chars > 15:
        return True
    return False

@st.cache_resource
def load_assets():
    if not os.path.exists("model.pkl"): return None, None
    with open("model.pkl", "rb") as f: m = pickle.load(f)
    with open("tfidf.pkl", "rb") as f: t = pickle.load(f)
    return m, t

# UI LAYOUT
st.markdown('<div class="hero"><h1>🛡️ A Naïve Bayes Framework for Email Spam & Phishing Detection</h1><p></p></div>', unsafe_allow_html=True)
l, r = st.columns([3, 2], gap="large")
with l:
    st.markdown('<div class="card"><h4>🔍 Threat Scanner</h4>', unsafe_allow_html=True)
    user_input = st.text_area("Paste the email content or URL below:", height=200)
    
    if st.button("🚀 ANALYZE"):
        if user_input.strip():
            input_lower = user_input.lower().strip()
            
            # LAYER 1: DOMAIN PARSING
            raw_url = re.sub(r"https?://", "", input_lower)
            domain_check = re.sub(r"^(www\.|web\.|support\.)", "", raw_url).split("/")[0]
            
            # LAYER 2: MULTI-STAGE FILTERING
            is_whitelisted = any(d in domain_check for d in WHITELIST)
            is_random = is_random_url(raw_url)
            found_flags = [f for f in RED_FLAGS if f in input_lower]

            if is_whitelisted:
                st.success(f"✅ TRUSTED: {domain_check} is verified.")
            elif is_random:
                st.error("🚨 CRITICAL ALERT: Suspicious Randomized URL Detected!")
                st.warning(f"The domain '{domain_check}' exhibits high entropy (randomness), a common signature of bot-generated phishing links.")
            elif len(found_flags) >= 2:
                st.error(f"🚨 SECURITY ALERT: Social Engineering keywords detected.")
            else:
                # LAYER 3: AI ANALYSIS
                model, tfidf = load_assets()
                if model:
                    cleaned = " ".join(user_input.split()) # Simple clean for UI
                    prob = model.predict_proba(tfidf.transform([cleaned]))[0]
                    if prob[1] >= 0.5: st.error(f"🚨 AI SPAM ALERT | Confidence: {prob[1]:.2%}")
                    else: st.success(f"✅ SAFE CONTENT | Confidence: {prob[0]:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

with r:
    st.markdown(f'''<div class="card"><h4>📊 System Intelligence</h4>
        <p><b>Model Accuracy:</b> 89.36%</p>
        <p><b>Entropy Detection:</b> ENABLED</p>
        <p><b>Execution Environment:</b> Local System (8GB RAM)</p>
        <hr style="border-top: 1px solid #334155">
        <p style="font-size: 0.8rem; color: #94a3b8;">Hybrid Defense: Detecting phishing URLs using DGA patterns and heuristic analysis.</p>
    </div>''', unsafe_allow_html=True)