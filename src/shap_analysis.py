"""
Steam Success Predictor - SHAP Model Interpretation
Day 3: Explainable AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_PATH = Path("models/catboost_baseline.pkl")
    DATA_PATH = Path("data/processed/steam_processed.csv")
    OUTPUT_DIR = Path("reports/shap_outputs")
    
    # Sample size for SHAP (larger = more accurate but slower)
    SHAP_SAMPLE_SIZE = 1000
    
    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {cls.OUTPUT_DIR}")


# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

def load_model_and_data(config: Config):
    """Load trained model and test data"""
    print("\n📊 Loading model and data...")
    
    # Load model
    model = CatBoostClassifier()
    model.load_model(str(config.MODEL_PATH))
    print(f"✓ Model loaded from: {config.MODEL_PATH}")
    
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    
    # Features (same as training)
    numeric_features = [
        'Price', 'Achievements', 'Required age', 'DiscountDLC count',
        'release_year', 'release_month', 'release_quarter'
    ]
    categorical_features = [
        'Developers', 'Publishers', 'Genres', 'Categories', 
        'Windows', 'Mac', 'Linux'
    ]
    
    available_num = [f for f in numeric_features if f in df.columns]
    available_cat = [f for f in categorical_features if f in df.columns]
    
    # Prepare features
    X = df[available_num + available_cat].copy()
    y = df['is_hit'].copy()
    
    # Fill missing values (same as training)
    for col in available_num:
        X[col] = X[col].fillna(X[col].median())
    for col in available_cat:
        X[col] = X[col].fillna('Unknown')
    
    # Sample for SHAP (faster computation)
    if len(X) > config.SHAP_SAMPLE_SIZE:
        X_sample = X.sample(config.SHAP_SAMPLE_SIZE, random_state=42)
        y_sample = y.loc[X_sample.index]
        print(f"✓ Sampled {config.SHAP_SAMPLE_SIZE} rows for SHAP")
    else:
        X_sample = X
        y_sample = y
    
    print(f"✓ Features: {len(available_num + available_cat)}")
    
    return model, X_sample, y_sample, available_cat


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(model, X: pd.DataFrame, y: pd.Series, 
                      cat_features: list, config: Config):
    """Run SHAP analysis and save visualizations"""
    print("\n" + "="*60)
    print("🔍 SHAP ANALYSIS")
    print("="*60)
    
    # Create CatBoost Pool
    pool = Pool(X, cat_features=cat_features)
    
    # Initialize SHAP explainer
    print("\n⏳ Initializing SHAP explainer (this may take 1-2 minutes)...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("⏳ Calculating SHAP values...")
    shap_values = explainer.shap_values(pool)
    print("✓ SHAP values calculated")
    
    # ========================================================================
    # 1. Summary Plot (Feature Importance)
    # ========================================================================
    print("\n📊 Generating summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X, 
        plot_type="bar", 
        show=False
    )
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '01_shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 01_shap_importance.png")
    
    # ========================================================================
    # 2. Summary Plot (Beeswarm - shows impact direction)
    # ========================================================================
    print("\n📊 Generating beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X, 
        show=False,
        color_bar_label="Feature Value"
    )
    plt.title('SHAP Beeswarm Plot (Feature Impact)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '02_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 02_shap_beeswarm.png")
    
    # ========================================================================
    # 3. Top Features DataFrame
    # ========================================================================
    print("\n📊 Generating feature impact DataFrame...")
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    shap_df.to_csv(config.OUTPUT_DIR / 'shap_feature_impact.csv', index=False)
    print(f"✓ Saved: shap_feature_impact.csv")
    
    # Print top 10
    print("\n🏆 Top 10 Features by SHAP Impact:")
    for i, row in shap_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    
   
    # ========================================================================
    # 4. Individual Prediction Analysis
    # ========================================================================
    print("\n📊 Analyzing individual predictions...")
    
    # Reset index for positional access (0-999)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    # Find positions in reset index
    hit_positions = np.where(y_reset == 1)[0]
    flop_positions = np.where(y_reset == 0)[0]
    
    hit_pos = hit_positions[0] if len(hit_positions) > 0 else 0
    flop_idx = flop_positions[0] if len(flop_positions) > 0 else 0
    
    # Create force plot for one example
    plt.figure(figsize=(10, 8))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[hit_pos],  # Use positional index
        X_reset.iloc[hit_pos], 
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Example (Position: {hit_pos})', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '03_shap_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 03_shap_example.png")
    
    if flop_idx:
        plt.figure(figsize=(10, 8))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[flop_idx], 
            X.loc[flop_idx], 
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - Flop Example (Actual: {y_sample.loc[flop_idx]})', 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / '04_shap_flop_example.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 04_shap_flop_example.png")
    
    # ========================================================================
    # 5. SHAP Insights Summary
    # ========================================================================
    print("\n💡 Generating insights...")
    
    insights = []
    insights.append("# SHAP Analysis Insights\n")
    insights.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    insights.append(f"Sample size: {len(X)}\n\n")
    
    insights.append("## Key Findings\n\n")
    insights.append("### Top Positive Factors (increase hit probability):\n")
    
    # Analyze direction of impact for top features
    for i, row in shap_df.head(5).iterrows():
        feature = row['feature']
        avg_shap = shap_values[:, X.columns.get_loc(feature)].mean()
        direction = "↑ increases" if avg_shap > 0 else "↓ decreases"
        insights.append(f"- **{feature}**: {direction} success (avg SHAP: {avg_shap:.4f})\n")
    
    insights.append("\n### Business Recommendations:\n")
    insights.append("1. Focus on popular genres (Genres is top feature)\n")
    insights.append("2. Consider price optimization (Price is top 5)\n")
    insights.append("3. Add achievements to increase engagement\n")
    insights.append("4. Partner with known publishers\n")
    
    insights_text = "".join(insights)
    
    with open(config.OUTPUT_DIR / 'shap_insights.md', 'w', encoding='utf-8') as f:
        f.write(insights_text)
    print(f"✓ Saved: shap_insights.md")
    
    print("\n" + insights_text)
    
    return shap_values, shap_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main SHAP analysis pipeline"""
    print("\n" + "🚀"*30)
    print("🎮 STEAM SUCCESS PREDICTOR - SHAP ANALYSIS")
    print("🚀"*30 + "\n")
    
    Config.setup()
    model, X, y, cat_features = load_model_and_data(Config)
    shap_values, shap_df = run_shap_analysis(model, X, y, cat_features, Config)
    
    print("\n" + "✅"*30)
    print("🎉 SHAP ANALYSIS COMPLETE!")
    print("✅"*30 + "\n")
    
    print("📁 OUTPUT FILES:")
    print(f"   - {Config.OUTPUT_DIR / '01_shap_importance.png'}")
    print(f"   - {Config.OUTPUT_DIR / '02_shap_beeswarm.png'}")
    print(f"   - {Config.OUTPUT_DIR / '03_shap_hit_example.png'}")
    print(f"   - {Config.OUTPUT_DIR / '04_shap_flop_example.png'}")
    print(f"   - {Config.OUTPUT_DIR / 'shap_feature_impact.csv'}")
    print(f"   - {Config.OUTPUT_DIR / 'shap_insights.md'}")
    print("\n📝 NEXT: Create Streamlit Demo\n")


if __name__ == "__main__":
    main()