"""
Steam Success Predictor - Streamlit Web Demo
Day 3: Interactive Prediction Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path("models/catboost_baseline.pkl")
THRESHOLD = 0.80  # Optimized threshold from Day 2

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎮 Steam Success Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model"""
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model

# ============================================================================
# SIDEBAR - INPUT FORM
# ============================================================================

def sidebar_input():
    """Create sidebar input form"""
    st.sidebar.header("🎮 Game Parameters")
    
    # Price
    price = st.sidebar.slider("💰 Price ($)", 0.0, 100.0, 29.99, 0.01)
    
    # Achievements
    achievements = st.sidebar.slider("🏆 Achievements", 0, 500, 50)
    
    # Required Age
    age_options = [0, 7, 12, 16, 18]
    required_age = st.sidebar.selectbox("🔞 Required Age", age_options, index=4)
    
    # DLC Count
    dlc_count = st.sidebar.slider("📦 DLC Count", 0, 100, 5)
    
    # Release Year
    release_year = st.sidebar.slider("📅 Release Year", 2010, 2024, 2023)
    
    # Release Month
    release_month = st.sidebar.slider("📅 Release Month", 1, 12, 11)
    
    # Release Quarter
    release_quarter = (release_month - 1) // 3 + 1
    
    # Genres
    genre_options = ['Action', 'Adventure', 'Casual', 'Indie', 'RPG', 'Simulation', 'Strategy', 'Sports', 'Racing']
    genres = st.sidebar.multiselect("🎭 Genres", genre_options, default=['Indie', 'Action'])
    
    # Categories
    category_options = ['Single-player', 'Multi-player', 'Co-op', 'Steam Achievements', 'Steam Cloud', 'VR Supported']
    categories = st.sidebar.multiselect("📋 Categories", category_options, default=['Single-player', 'Steam Achievements'])
    
    # Platforms
    windows = st.sidebar.checkbox("💻 Windows", value=True)
    mac = st.sidebar.checkbox("🍎 Mac", value=False)
    linux = st.sidebar.checkbox("🐧 Linux", value=False)
    
    # Developers & Publishers (simplified - top ones)
    developer_options = ['Valve', 'Ubisoft', 'EA', 'Indie Studio', 'Square Enix', 'Bethesda', 'Activision', 'Other']
    developer = st.sidebar.selectbox("🏢 Developer", developer_options)
    
    publisher_options = ['Valve', 'Ubisoft', 'EA', 'Indie Publishing', 'Square Enix', 'Bethesda', 'Activision', 'Other']
    publisher = st.sidebar.selectbox("📦 Publisher", publisher_options)
    
    # Compile input
    input_data = {
        'Price': [price],
        'Achievements': [achievements],
        'Required age': [required_age],
        'DiscountDLC count': [dlc_count],
        'release_year': [release_year],
        'release_month': [release_month],
        'release_quarter': [release_quarter],
        'Developers': [developer],
        'Publishers': [publisher],
        'Genres': [', '.join(genres) if genres else 'Unknown'],
        'Categories': [', '.join(categories) if categories else 'Unknown'],
        'Windows': [windows],
        'Mac': [mac],
        'Linux': [linux]
    }
    
    return pd.DataFrame(input_data)

# ============================================================================
# PREDICTION
# ============================================================================

def predict_success(model, input_df, threshold=THRESHOLD):
    """Make prediction"""
    # Get prediction
    proba = model.predict_proba(input_df)[0][1]
    prediction = "🏆 HIT" if proba >= threshold else "💀 FLOP"
    
    return proba, prediction

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🎮 Steam Success Predictor")
    st.markdown("""
    ### Predict if your game will be a **Hit** or **Flop** on Steam!
    
    This ML model uses **CatBoost** trained on 122,611 Steam games to predict success 
    based on pre-launch features only.
    
    **Model Performance:**
    - 📈 ROC-AUC: **0.93**
    - 🎯 Recall: **81%** (finds most hits)
    - ⚖️ F1-Score: **0.52** (optimized threshold)
    """)
    
    st.divider()
    
    # Sidebar input
    input_df = sidebar_input()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Game Configuration")
        st.dataframe(input_df.T, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔮 Predict Success", type="primary", use_container_width=True):
            model = load_model()
            proba, prediction = predict_success(model, input_df)
            
            # Display prediction
            if prediction == "🏆 HIT":
                st.success(f"# {prediction}")
                st.metric("Success Probability", f"{proba:.2%}")
                st.info("✅ High chance of success! Consider greenlighting this game.")
            else:
                st.error(f"# {prediction}")
                st.metric("Success Probability", f"{proba:.2%}")
                st.warning("⚠️ Lower chance of success. Review game features before investing.")
            
            # Progress bar
            st.progress(proba)
            
            # Threshold info
            st.caption(f"📌 Threshold: {THRESHOLD:.2f} (optimized for F1-score)")
    
    st.divider()
    
    # Model Info
    st.subheader("ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", "122,611")
    
    with col2:
        st.metric("Features", "14")
    
    with col3:
        st.metric("Model", "CatBoost (GPU)")
    
    # Feature Importance
    st.subheader("🏆 Top Features (SHAP Analysis)")
    st.markdown("""
    | Feature | SHAP Impact |
    |---------|-------------|
    | Categories | 0.661 |
    | Price | 0.592 |
    | Achievements | 0.522 |
    | Genres | 0.520 |
    | DLC Count | 0.493 |
    | Publishers | 0.445 |
    | Mac | 0.314 |
    | Required Age | 0.304 |
    | Developers | 0.285 |
    | Linux | 0.040 |
    """)
    
    # Disclaimer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This is a demo project for educational purposes. 
    Predictions are based on historical data and should not be used as 
    the sole basis for investment decisions.
    
    📁 **GitHub:** [Your Repository Link]
    """)


if __name__ == "__main__":
    main()