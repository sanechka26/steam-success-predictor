"""
Steam Success Predictor - Model Training Pipeline
Day 2: Feature Engineering + CatBoost Baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from catboost import CatBoostClassifier, Pool
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    DATA_PATH = Path("data/processed/steam_processed.csv")
    MODEL_PATH = Path("models/catboost_baseline.pkl")
    REPORTS_PATH = Path("reports")
    
    # Features to use
    NUMERIC_FEATURES = [
        'Price', 
        'Achievements',
        'Required age', 
        'DiscountDLC count',
        'release_year', 
        'release_month', 
        'release_quarter',
        'release_day_of_week'  # Можно добавить
    ]

    CATEGORICAL_FEATURES = [
        'Developers', 
        'Publishers', 
        'Genres', 
        'Categories', 
        'Windows', 
        'Mac', 
        'Linux',
        'VR Supported'  # Если есть
    ]
    
    TARGET = 'is_hit'
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ITERATIONS = 10000
    LEARNING_RATE = 0.01
    DEPTH = 10
    L2_REG = 10
    CV_FOLDS = 5  # 5-Fold Cross-Validation
    
    @classmethod
    def setup(cls):
        """Create directories"""
        cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"✓ Model will be saved to: {cls.MODEL_PATH}")


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load processed data"""
    print("\n📊 Loading data...")
    df = pd.read_csv(path)
    print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def preprocess_data(df: pd.DataFrame, config: Config) -> tuple:
    """Prepare features and target"""
    print("\n🔧 Preprocessing data...")
    
    df = df.copy()
    
    # Fill missing values for numeric features
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values for categorical features
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
    
    # Select features that exist in dataframe
    available_num = [f for f in config.NUMERIC_FEATURES if f in df.columns]
    available_cat = [f for f in config.CATEGORICAL_FEATURES if f in df.columns]
    
    print(f"✓ Numeric features: {len(available_num)}")
    print(f"✓ Categorical features: {len(available_cat)}")
    
    X = df[available_num + available_cat]
    y = df[config.TARGET]
    
    return X, y, available_num, available_cat


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X: pd.DataFrame, y: pd.Series, num_features: list, 
                cat_features: list, config: Config) -> CatBoostClassifier:
    """Train CatBoost model"""
    print("\n🤖 Training CatBoost model...")
    
    # Split data with stratification (important for imbalanced data!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y  # Keep class distribution
    )
    
    print(f"✓ Train size: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✓ Test size: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"✓ Train hit rate: {y_train.mean()*100:.2f}%")
    print(f"✓ Test hit rate: {y_test.mean()*100:.2f}%")
    
    # Create CatBoost pools
    train_pool = Pool(
        X_train, 
        y_train, 
        cat_features=cat_features
    )
    
    test_pool = Pool(
        X_test, 
        y_test, 
        cat_features=cat_features
    )
    
    # Initialize model with class weights for imbalanced data
    model = CatBoostClassifier(
        iterations=config.ITERATIONS,
        learning_rate=config.LEARNING_RATE,
        depth=config.DEPTH,
        l2_leaf_reg=config.L2_REG,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=config.RANDOM_STATE,
        verbose=100,
        auto_class_weights='Balanced',
        # early_stopping_rounds=100,
    
    # 🔥 GPU SETTINGS
        task_type='GPU',
        devices='0',  # Use first GPU (your RTX 5070 Ti)
        gpu_ram_part=0.95  # Use 95% of GPU memory
    )
    
    # Train
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True
    )
    
    print("✓ Training complete!")
    
    return model, X_test, y_test


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model: CatBoostClassifier, X_test: pd.DataFrame, 
                   y_test: pd.Series, config: Config):
    """Evaluate model and save reports"""
    print("\n" + "="*60)
    print("📈 MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    optimal_thresh, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    print(f"\n🎯 Optimal threshold: {optimal_thresh:.2f} (F1: {best_f1:.4f})")
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n🎯 Key Metrics:")
    print(f"  • ROC-AUC:  {roc_auc:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Flop', 'Hit']))
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[{cm[0,0]:,}  {cm[0,1]:,}]\n   [{cm[1,0]:,}  {cm[1,1]:,}]]")
    print(f"   [TN    FP]\n   [FN    TP]")
    
    # Save metrics to file
    metrics = {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = config.REPORTS_PATH / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Steam Success Predictor - Model Metrics\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    return metrics


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(model: CatBoostClassifier, X_train: pd.DataFrame, 
                           config: Config):
    """Get and save feature importance"""
    print("\n" + "="*60)
    print("🏆 FEATURE IMPORTANCE (Top 15)")
    print("="*60)
    
    importance = model.get_feature_importance()
    feature_names = X_train.columns.tolist()
    
    # Create dataframe
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Print top 15
    for i, row in fi_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save to file
    fi_df.to_csv(config.REPORTS_PATH / 'feature_importance.csv', index=False)
    print(f"\n✓ Feature importance saved to: {config.REPORTS_PATH / 'feature_importance.csv'}")
    
    return fi_df


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model: CatBoostClassifier, config: Config):
    """Save trained model"""
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    model.save_model(str(config.MODEL_PATH))
    print(f"✓ Model saved to: {config.MODEL_PATH}")


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find threshold that maximizes F1 or other metric"""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        else:
            score = recall_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "🚀"*30)
    print("🎮 STEAM SUCCESS PREDICTOR - TRAINING")
    print("🚀"*30 + "\n")
    
    # Setup
    Config.setup()
    
    # Load data
    df = load_data(Config.DATA_PATH)
    
    # Preprocess
    X, y, num_features, cat_features = preprocess_data(df, Config)
    
    # Train
    model, X_test, y_test = train_model(X, y, num_features, cat_features, Config)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, Config)
    
    # Feature importance
    get_feature_importance(model, X, Config)
    
    # Save model
    save_model(model, Config)
    
    print("\n" + "✅"*30)
    print("🎉 TRAINING COMPLETE!")
    print("✅"*30 + "\n")
    
    # Summary
    print("📁 OUTPUT FILES:")
    print(f"   - {Config.MODEL_PATH}")
    print(f"   - {Config.REPORTS_PATH / 'metrics.txt'}")
    print(f"   - {Config.REPORTS_PATH / 'feature_importance.csv'}")
    print("\n📝 NEXT: Day 3 - Model Interpretation (SHAP) + API\n")


if __name__ == "__main__":
    main()