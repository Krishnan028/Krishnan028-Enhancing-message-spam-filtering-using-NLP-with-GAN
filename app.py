import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

from models.gan_text_detector import get_model as get_text_model
from models.image_spam_detector import get_model as get_image_model
from models.model_trainer import get_trainer
from utils.ocr_processor import get_ocr_processor
from utils.video_processor import get_video_processor
from utils.sample_data import get_sample_messages
from utils.analytics import get_analytics_tracker
from utils.batch_processor import get_batch_processor
from utils.pdf_reporter import get_pdf_reporter

# Page configuration
st.set_page_config(
    page_title="SMS Spam Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI/UX
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom Card Styling */
    .stCard {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        text-align: center;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Status indicators */
    .status-spam {
        color: #ef4444;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background-color: #fee2e2;
    }
    
    .status-ham {
        color: #10b981;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background-color: #d1fae5;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize models and utilities
@st.cache_resource
def load_models():
    text_detector = get_text_model()
    image_detector = get_image_model()
    ocr_processor = get_ocr_processor()
    video_processor = get_video_processor()
    trainer = get_trainer(text_detector, image_detector)
    batch_processor = get_batch_processor(text_detector, image_detector, ocr_processor)
    pdf_reporter = get_pdf_reporter()
    return text_detector, image_detector, ocr_processor, video_processor, trainer, batch_processor, pdf_reporter

text_detector, image_detector, ocr_processor, video_processor, trainer, batch_processor, pdf_reporter = load_models()

# Initialize analytics tracker in session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = get_analytics_tracker()

def main():
    # Sidebar Branding
    with st.sidebar:
        st.markdown("<h1 style='color: #3b82f6;'>🛡️ SMS Guard AI</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.info("Multi-modal SMS spam detection system using GANs and CNNs.")
        
        # Quick Statistics in Sidebar
        stats = st.session_state.analytics.get_stats()
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Total", stats['total'])
        col2.metric("Spam Rate", f"{stats['spam_rate']:.1f}%")
        
        st.markdown("---")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        
        if st.button("🗑️ Clear History"):
            st.session_state.analytics.clear_history()
            st.rerun()

    # Main Header Area
    st.markdown("<h1 class='main-header'>Advanced SMS Spam Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Detect spam across text, images, and video using state-of-the-art Generative Adversarial Networks.</p>", unsafe_allow_html=True)

    # Main Navigation Tabs
    tabs = st.tabs([
        "📱 Text Analysis", 
        "🖼️ Image Analysis", 
        "🎥 Video Analysis", 
        "📚 Sample Dataset",
        "🎓 Model Training",
        "📦 Batch Process",
        "📈 Analytics",
        "📊 Performance",
        "ℹ️ About"
    ])

    # 1. Text Analysis Tab
    with tabs[0]:
        st.markdown("### 📱 Single Message Analysis")
        input_text = st.text_area("Enter message content here:", height=150, placeholder="e.g., Congratulations! You've won a free gift card. Click here to claim now!")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_btn = st.button("🔍 Analyze Message", width="stretch", type="primary")

        if analyze_btn:
            if input_text:
                with st.spinner("🔍 NLP based Analyzing..."):
                    score = text_detector.predict(input_text)
                    is_spam = score >= confidence_threshold
                    
                    # Log result
                    st.session_state.analytics.log_detection("Text", "Text Input", is_spam, score)
                    
                    # Display Results
                    st.markdown("---")
                    res_col1, res_col2 = st.columns([2, 3])
                    
                    with res_col1:
                        if is_spam:
                            st.markdown("#### Result: <span class='status-spam'>SPAM DETECTED</span>", unsafe_allow_html=True)
                            st.error("This message shows high probability of being spam/phishing.")
                        else:
                            st.markdown("#### Result: <span class='status-ham'>LEGITIMATE</span>", unsafe_allow_html=True)
                            st.success("This message appears to be safe and legitimate.")
                    
                    with res_col2:
                        st.write(f"**Confidence Score:** {score:.2%}")
                        st.progress(score)
            else:
                st.warning("Please enter some text to analyze.")

    # 2. Image Analysis Tab
    with tabs[1]:
        st.markdown("### 🖼️ Visual Content Analysis")
        uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_col1, img_col2 = st.columns([1, 1])
            with img_col1:
                st.image(image, caption="Uploaded Content", width="stretch")
            
            with img_col2:
                if st.button("🚀 Analyze Image", key="img_analyze", type="primary"):
                    with st.spinner("🧠 Performing Multi-modal Analysis..."):
                        img_spam_prob, features = image_detector.predict(image)
                        ocr_result = ocr_processor.extract_text(image)
                        ocr_text = ocr_result.get('text', '').strip()
                        text_score = text_detector.predict(ocr_text) if ocr_text else 0.0

                        import re as _re

                        # --- Multi-signal spam detection on extracted text ---
                        has_url = bool(_re.search(r'https?://|www\.|\.com|\.co\.uk|\.net|\.org|\.info|\.xyz', ocr_text, _re.IGNORECASE))
                        has_phone = bool(_re.search(r'\b(\+?\d[\d\s\-]{7,15})\b', ocr_text))
                        has_money = bool(_re.search(r'[£$€₹]\s*\d+|free\s|voucher|coupon|gift\s*card', ocr_text, _re.IGNORECASE))
                        has_urgency = bool(_re.search(r'hurry|limited|act now|expires|last chance|immediately|urgent|congratulations|winner|won\b', ocr_text, _re.IGNORECASE))
                        has_cta = bool(_re.search(r'click|tap|go here|visit|call now|reply|claim|get it', ocr_text, _re.IGNORECASE))

                        # Count how many spam signals are present
                        spam_signals = sum([has_url, has_phone, has_money, has_urgency, has_cta])

                        if ocr_text and len(ocr_text) > 10:
                            # OCR found meaningful text → text analysis is the primary signal
                            final_score = 0.80 * text_score + 0.20 * img_spam_prob
                        else:
                            final_score = img_spam_prob

                        # Boost score based on spam signals found in image text
                        if spam_signals >= 3:
                            final_score = max(final_score, 0.92)
                        elif spam_signals >= 2:
                            final_score = max(final_score, 0.82)

                        # Hard override: URL detected in image text → very likely spam
                        if has_url:
                            final_score = max(final_score, 0.88)

                        # Money/voucher + CTA = scam pattern
                        if has_money and has_cta:
                            final_score = max(final_score, 0.90)

                        is_spam = final_score >= confidence_threshold
                        st.session_state.analytics.log_detection("Image", uploaded_file.name, is_spam, final_score)

                        if is_spam:
                            st.markdown("#### Result: <span class='status-spam'>SPAM DETECTED</span>", unsafe_allow_html=True)
                            st.error("This image contains spam indicators.")
                        else:
                            st.markdown("#### Result: <span class='status-ham'>LEGITIMATE</span>", unsafe_allow_html=True)
                            st.success("This image appears safe and legitimate.")

                        st.metric("Total Confidence", f"{final_score:.1%}")
                        st.progress(final_score)

                        # Show detected signals
                        if spam_signals > 0:
                            signals_found = []
                            if has_url: signals_found.append("🔗 URL/Link")
                            if has_phone: signals_found.append("📞 Phone Number")
                            if has_money: signals_found.append("💰 Money/Voucher")
                            if has_urgency: signals_found.append("⚡ Urgency Words")
                            if has_cta: signals_found.append("👆 Call-to-Action")
                            st.warning(f"**Spam signals detected:** {', '.join(signals_found)}")

                        if ocr_text:
                            with st.expander("📝 Extracted Text (OCR)", expanded=True):
                                st.write(ocr_text)
                                st.caption(f"OCR Confidence: {ocr_result.get('confidence', 0):.1f}% | Words found: {ocr_result.get('word_count', 0)}")
                                st.caption(f"Text spam score: {text_score:.2%} | Image CNN score: {img_spam_prob:.2%}")


    # 3. Video Analysis Tab
    with tabs[2]:
        st.markdown("### 🎥 Video Message Analysis")
        uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'mov', 'avi'])
        
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                vpath = tfile.name
            
            st.video(vpath)
            if st.button("🎬 Analyze Video", type="primary"):
                with st.spinner("🎞️ Analyzing frames..."):
                    frames, nums = video_processor.extract_frames(vpath)
                    results = video_processor.analyze_video_frames(frames, image_detector, text_detector, ocr_processor)
                    overall_score = (results['avg_image_spam_score'] + results['avg_text_spam_score']) / 2
                    is_spam = overall_score >= confidence_threshold
                    
                    st.session_state.analytics.log_detection("Video", uploaded_video.name, is_spam, overall_score)
                    
                    if is_spam:
                        st.markdown("#### Result: <span class='status-spam'>SPAM DETECTED</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("#### Result: <span class='status-ham'>LEGITIMATE</span>", unsafe_allow_html=True)
                    st.metric("Overall Score", f"{overall_score:.1%}")

    # 4. Sample Dataset Tab
    with tabs[3]:
        st.markdown("### 📚 Sample SMS Dataset")
        samples = get_sample_messages()
        cat = st.radio("Category:", ["Spam", "Legitimate"], horizontal=True)
        
        target_list = samples['spam'] if cat == "Spam" else samples['legitimate']
        for i, text in enumerate(target_list):
            with st.expander(f"Sample {i+1}: {text[:50]}..."):
                st.write(text)
                if st.button("Analyze", key=f"s_{cat}_{i}"):
                    score = text_detector.predict(text)
                    st.metric("Spam Score", f"{score:.1%}")

    # 5. Model Training Tab
    with tabs[4]:
        st.markdown("### 🎓 Model Training")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text GAN")
            if st.button("Train Text Model"):
                with st.spinner("Training..."):
                    h = trainer.train_text_model(["Sample text"], [1], epochs=5)
                    st.success("Done!")
        with col2:
            st.subheader("Image CNN")
            if st.button("Train Image Model"):
                with st.spinner("Training..."):
                    h = trainer.train_image_model([], [], epochs=5)
                    st.success("Done!")

    # 6. Batch Processing Tab
    with tabs[5]:
        st.markdown("### 📦 Batch Processing")
        raw = st.text_area("Enter messages (one per line):")
        if st.button("Process Batch", type="primary"):
            msgs = [m.strip() for m in raw.split('\n') if m.strip()]
            if msgs:
                res = []
                for m in msgs:
                    s = text_detector.predict(m)
                    res.append({"Message": m[:50], "Score": f"{s:.2f}", "Status": "SPAM" if s > 0.5 else "HAM"})
                st.table(res)

    # 7. Analytics Dashboard Tab
    with tabs[6]:
        st.markdown("### 📈 Analytics")
        hist = st.session_state.analytics.get_history_df()
        if not hist.empty:
            st.dataframe(hist, width="stretch")
            st.pyplot(st.session_state.analytics.generate_summary_plot())
        else:
            st.info("No data yet.")

    # 8. Performance Tab
    with tabs[7]:
        st.markdown("### 📊 Performance")
        st.write("GAN Discriminator: BiLSTM Architecture")
        st.write("Image CNN: 3 Convolutional Blocks")

    # 9. About Tab
    with tabs[8]:
        st.markdown("### ℹ️ About")
        st.write("SMS Spam Guard AI - Final Year Project 2026")

if __name__ == "__main__":
    main()