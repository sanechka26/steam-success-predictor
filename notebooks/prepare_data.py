"""
Steam Data Preparation - No plots, just data processing
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

RAW_PATH = Path("data/raw/steam.csv")
PROCESSED_PATH = Path("data/processed/steam_processed.csv")

SUCCESS_REVIEWS_THRESHOLD = 1000
SUCCESS_POSITIVE_RATE = 70

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n🎮 STEAM DATA PREPARATION\n")
    
    # Create directories
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv(RAW_PATH)
    print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Create target variable
    print("\n🎯 Creating target variable...")
    df['total_reviews'] = df['Positive'] + df['Negative']
    df['positive_rate'] = (df['Positive'] / df['total_reviews'].replace(0, 1) * 100).round(2)
    
    df['is_hit'] = (
        (df['total_reviews'] >= SUCCESS_REVIEWS_THRESHOLD) & 
        (df['positive_rate'] >= SUCCESS_POSITIVE_RATE)
    ).astype(int)
    
    hit_count = df['is_hit'].sum()
    hit_pct = hit_count / len(df) * 100
    
    print(f"✓ Hits: {hit_count:,} ({hit_pct:.2f}%)")
    print(f"✓ Flops: {len(df) - hit_count:,} ({100-hit_pct:.2f}%)")
    
    # Extract date features
    print("\n📅 Extracting date features...")
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['release_year'] = df['Release date'].dt.year
    df['release_month'] = df['Release date'].dt.month
    df['release_quarter'] = df['Release date'].dt.quarter
    
    # Save processed data
    print(f"\n💾 Saving to {PROCESSED_PATH}...")
    df.to_csv(PROCESSED_PATH, index=False)
    print("✓ Done!")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\n📁 Output: {PROCESSED_PATH.absolute()}")
    print("\n📝 NEXT: Run Day 2 - Feature Engineering & Model\n")

if __name__ == "__main__":
    main()