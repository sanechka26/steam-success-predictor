"""
Steam Games EDA - Exploratory Data Analysis
Day 1 of Steam Success Predictor Project
"""
import matplotlib
matplotlib.use('Agg')  # <-- ВАЖНО: Отключает всплывающие окна
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for EDA"""
    RAW_DATA_PATH = Path("data/raw/steam.csv")
    OUTPUT_DIR = Path("notebooks/eda_outputs")
    FIG_SIZE = (14, 10)
    RANDOM_STATE = 42
    
    # Success criteria
    SUCCESS_REVIEWS_THRESHOLD = 1000  # Min reviews to be considered
    SUCCESS_POSITIVE_RATE = 70  # Min positive rate %
    
    # Column mappings for THIS dataset
    COL_POSITIVE = 'Positive'
    COL_NEGATIVE = 'Negative'
    COL_DEVELOPERS = 'Developers'
    COL_PUBLISHERS = 'Publishers'
    COL_RELEASE_DATE = 'Release date'
    COL_PRICE = 'Price'
    COL_GENRES = 'Genres'
    COL_TAGS = 'Tags'
    COL_CATEGORIES = 'Categories'
    
    @classmethod
    def setup(cls):
        """Create output directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        print(f"✓ Output directory: {cls.OUTPUT_DIR.absolute()}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load and return dataframe with basic info"""
    print("\n" + "="*60)
    print("📊 LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(path)
    
    print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


# ============================================================================
# BASIC ANALYSIS
# ============================================================================

def basic_info(df: pd.DataFrame) -> dict:
    """Collect basic dataframe information"""
    print("\n" + "="*60)
    print("📋 BASIC INFORMATION")
    print("="*60)
    
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    print(f"\nColumns ({len(info['columns'])}):")
    for i, col in enumerate(info['columns'], 1):
        print(f"  {i}. {col}")
    
    print(f"\nData Types:")
    for dtype, count in pd.Series(info['dtypes']).value_counts().items():
        print(f"  {dtype}: {count}")
    
    print(f"\nMissing Values (top 10):")
    missing_sorted = sorted(info['missing_pct'].items(), key=lambda x: x[1], reverse=True)[:10]
    for col, pct in missing_sorted:
        if pct > 0:
            print(f"  {col}: {pct}%")
    
    print(f"\nDuplicates: {info['duplicates']} ({info['duplicates']/len(df)*100:.2f}%)")
    
    return info


# ============================================================================
# TARGET VARIABLE CREATION
# ============================================================================

def create_target_variable(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Create binary target variable: Hit (1) or Flop (0)"""
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE CREATION")
    print("="*60)
    
    df = df.copy()
    
    # Calculate total reviews and positive rate
    df['total_reviews'] = df[config.COL_POSITIVE] + df[config.COL_NEGATIVE]
    df['positive_rate'] = (df[config.COL_POSITIVE] / df['total_reviews'].replace(0, 1) * 100).round(2)
    
    print(f"✓ Calculated 'total_reviews' = {config.COL_POSITIVE} + {config.COL_NEGATIVE}")
    print(f"✓ Calculated 'positive_rate' from {config.COL_POSITIVE}/Total")
    
    # Create target based on reviews and positive rate
    df['is_hit'] = (
        (df['total_reviews'] >= config.SUCCESS_REVIEWS_THRESHOLD) & 
        (df['positive_rate'] >= config.SUCCESS_POSITIVE_RATE)
    ).astype(int)
    
    # Statistics
    hit_count = df['is_hit'].sum()
    hit_pct = hit_count / len(df) * 100
    
    print(f"\nTarget Distribution:")
    print(f"  🏆 Hits (1): {hit_count:,} ({hit_pct:.2f}%)")
    print(f"  💀 Flops (0): {len(df) - hit_count:,} ({100-hit_pct:.2f}%)")
    print(f"\nCriteria: Reviews ≥ {config.SUCCESS_REVIEWS_THRESHOLD} AND Positive Rate ≥ {config.SUCCESS_POSITIVE_RATE}%")
    
    return df


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_target_distribution(df: pd.DataFrame, config: Config):
    """Plot target variable distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2ecc71', '#e74c3c']
    axes[0].pie(
        df['is_hit'].value_counts(),
        labels=['Flop', 'Hit'],
        autopct='%1.1f%%',
        colors=colors,
        explode=(0.05, 0.05)
    )
    axes[0].set_title('Target Distribution (Hit vs Flop)', fontsize=14, fontweight='bold')
    
    sns.countplot(data=df, x='is_hit', ax=axes[1], palette=colors)
    axes[1].set_title('Count Plot', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Is Hit')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved: 01_target_distribution.png")
    #plt.show()


def plot_numeric_distributions(df: pd.DataFrame, config: Config):
    """Plot distributions of key numeric features"""
    numeric_cols = [config.COL_PRICE, 'total_reviews', 'positive_rate', 'Achievements']
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if not available_cols:
        print("⚠ No numeric columns found for distribution plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(available_cols[:4]):
        ax = axes[idx]
        sns.histplot(data=df, x=col, kde=True, ax=ax, color='#3498db')
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
    
    for idx in range(len(available_cols), 4):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '02_numeric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved: 02_numeric_distributions.png")
    #plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, config: Config):
    """Plot correlation heatmap for numeric features"""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠ Not enough numeric columns for correlation")
        return
    
    corr = numeric_df.corr()
    
    # Find top correlations with target
    if 'is_hit' in corr.columns:
        target_corr = corr['is_hit'].drop('is_hit').abs().sort_values(ascending=False)
        print("\nTop correlations with 'is_hit':")
        for col, val in target_corr.head(5).items():
            print(f"  {col}: {val:.3f}")
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Correlation Heatmap - Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved: 03_correlation_heatmap.png")
    #plt.show()


def plot_temporal_analysis(df: pd.DataFrame, config: Config):
    """Analyze release dates and seasonality"""
    print("\n" + "="*60)
    print("📅 TEMPORAL ANALYSIS")
    print("="*60)
    
    df = df.copy()
    df[config.COL_RELEASE_DATE] = pd.to_datetime(df[config.COL_RELEASE_DATE], errors='coerce')
    df = df.dropna(subset=[config.COL_RELEASE_DATE])
    
    df['release_year'] = df[config.COL_RELEASE_DATE].dt.year
    df['release_month'] = df[config.COL_RELEASE_DATE].dt.month
    df['release_quarter'] = df[config.COL_RELEASE_DATE].dt.quarter
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Games per year
    yearly = df.groupby('release_year').size()
    axes[0, 0].plot(yearly.index, yearly.values, marker='o', linewidth=2, color='#9b59b6')
    axes[0, 0].set_title('Games Released per Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Hit rate per year
    yearly_hit = df.groupby('release_year')['is_hit'].mean() * 100
    axes[0, 1].bar(yearly_hit.index, yearly_hit.values, color='#2ecc71')
    axes[0, 1].set_title('Hit Rate per Year (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Hit Rate %')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Games per month
    monthly = df.groupby('release_month').size()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].bar(monthly.index, monthly.values, color='#3498db')
    axes[1, 0].set_title('Games Released per Month', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names)
    
    # Hit rate per quarter
    quarterly = df.groupby('release_quarter')['is_hit'].mean() * 100
    axes[1, 1].bar(quarterly.index, quarterly.values, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    axes[1, 1].set_title('Hit Rate per Quarter (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Quarter')
    axes[1, 1].set_ylabel('Hit Rate %')
    axes[1, 1].set_xticks([1, 2, 3, 4])
    axes[1, 1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '04_temporal_analysis.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved: 04_temporal_analysis.png")
    #plt.show()
    
    return df


def plot_categorical_analysis(df: pd.DataFrame, config: Config):
    """Analyze categorical features (genres, developers, etc.)"""
    print("\n" + "="*60)
    print("🏷️ CATEGORICAL ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top 10 developers
    if config.COL_DEVELOPERS in df.columns:
        top_dev = df[config.COL_DEVELOPERS].value_counts().head(10)
        axes[0, 0].barh(range(len(top_dev)), top_dev.values, color='#9b59b6')
        axes[0, 0].set_yticks(range(len(top_dev)))
        axes[0, 0].set_yticklabels(top_dev.index)
        axes[0, 0].set_title('Top 10 Developers by Game Count', fontsize=12, fontweight='bold')
        axes[0, 0].invert_yaxis()
    
    # Top 10 publishers
    if config.COL_PUBLISHERS in df.columns:
        top_pub = df[config.COL_PUBLISHERS].value_counts().head(10)
        axes[0, 1].barh(range(len(top_pub)), top_pub.values, color='#e67e22')
        axes[0, 1].set_yticks(range(len(top_pub)))
        axes[0, 1].set_yticklabels(top_pub.index)
        axes[0, 1].set_title('Top 10 Publishers by Game Count', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
    
    # Price distribution by hit/flop
    if config.COL_PRICE in df.columns:
        sns.boxplot(data=df, x='is_hit', y=config.COL_PRICE, ax=axes[1, 0], palette=['#e74c3c', '#2ecc71'])
        axes[1, 0].set_title('Price Distribution by Hit/Flop', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Is Hit')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].set_xticklabels(['Flop', 'Hit'])
    
    # Reviews distribution by hit/flop
    if 'total_reviews' in df.columns:
        sns.boxplot(data=df, x='is_hit', y='total_reviews', ax=axes[1, 1], palette=['#e74c3c', '#2ecc71'])
        axes[1, 1].set_title('Reviews Distribution by Hit/Flop', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Is Hit')
        axes[1, 1].set_ylabel('Total Reviews')
        axes[1, 1].set_xticklabels(['Flop', 'Hit'])
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '05_categorical_analysis.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved: 05_categorical_analysis.png")
    # plt.show()


# ============================================================================
# INSIGHTS GENERATION
# ============================================================================

def generate_insights(df: pd.DataFrame, info: dict, config: Config) -> str:
    """Generate text insights from EDA"""
    print("\n" + "="*60)
    print("💡 GENERATING INSIGHTS")
    print("="*60)
    
    insights = []
    insights.append(f"# EDA Insights - Steam Games Dataset")
    insights.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    insights.append("")
    
    # Dataset info
    insights.append(f"## Dataset Overview")
    insights.append(f"- **Total Games:** {df.shape[0]:,}")
    insights.append(f"- **Features:** {df.shape[1]}")
    insights.append(f"- **Duplicates:** {info['duplicates']} ({info['duplicates']/len(df)*100:.2f}%)")
    insights.append("")
    
    # Target distribution
    hit_rate = df['is_hit'].mean() * 100
    insights.append(f"## Target Variable (Hit/Flop)")
    insights.append(f"- **Hit Rate:** {hit_rate:.2f}%")
    insights.append(f"- **Criteria:** Reviews ≥ {config.SUCCESS_REVIEWS_THRESHOLD} AND Positive Rate ≥ {config.SUCCESS_POSITIVE_RATE}%")
    insights.append(f"- **Class Balance:** {'Balanced' if 30 < hit_rate < 70 else 'Imbalanced'}")
    insights.append("")
    
    # Missing values
    high_missing = [col for col, pct in info['missing_pct'].items() if pct > 30]
    if high_missing:
        insights.append(f"## High Missing Values (>30%)")
        for col in high_missing[:5]:
            insights.append(f"- {col}: {info['missing_pct'][col]}%")
        insights.append("")
    
    # Key findings
    insights.append(f"## Key Findings & Hypotheses")
    insights.append(f"1. Price range $20-40 may have optimal success rate")
    insights.append(f"2. Q4 releases (Oct-Dec) might have higher hit rates")
    insights.append(f"3. Known developers/publishers have advantage")
    insights.append(f"4. Multi-player games may perform better")
    insights.append("")
    
    insights.append(f"## Next Steps")
    insights.append(f"1. Handle missing values (impute or drop)")
    insights.append(f"2. Feature engineering (year, month, price bins)")
    insights.append(f"3. Encode categorical variables")
    insights.append(f"4. Build baseline model")
    
    # Save insights
    insights_text = "\n".join(insights)
    insights_path = config.OUTPUT_DIR / 'insights.md'
    with open(insights_path, 'w', encoding='utf-8') as f:
        f.write(insights_text)
    print(f"✓ Saved: insights.md")
    
    print("\n" + insights_text)
    
    return insights_text


# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================

def save_processed_data(df: pd.DataFrame, config: Config):
    """Save processed dataframe for next steps"""
    output_path = config.OUTPUT_DIR.parent.parent / 'data' / 'processed' / 'steam_processed.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved processed data: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main EDA pipeline"""
    print("\n" + "🚀"*30)
    print("🎮 STEAM GAMES EDA PIPELINE")
    print("🚀"*30 + "\n")
    
    # Setup
    Config.setup()
    
    # Load data
    df = load_data(Config.RAW_DATA_PATH)
    
    # Basic info
    info = basic_info(df)
    
    # Create target
    df = create_target_variable(df, Config)
    
    # Visualizations
    plot_target_distribution(df, Config)
    plot_numeric_distributions(df, Config)
    plot_correlation_heatmap(df, Config)
    plot_temporal_analysis(df, Config)
    plot_categorical_analysis(df, Config)
    
    # Generate insights
    generate_insights(df, info, Config)
    
    # Save processed data
    save_processed_data(df, Config)
    
    print("\n" + "✅"*30)
    print("🎉 EDA COMPLETED SUCCESSFULLY!")
    print("✅"*30 + "\n")
    
    # Summary for user
    print("📁 OUTPUT FILES:")
    print(f"   - {Config.OUTPUT_DIR / '01_target_distribution.png'}")
    print(f"   - {Config.OUTPUT_DIR / '02_numeric_distributions.png'}")
    print(f"   - {Config.OUTPUT_DIR / '03_correlation_heatmap.png'}")
    print(f"   - {Config.OUTPUT_DIR / '04_temporal_analysis.png'}")
    print(f"   - {Config.OUTPUT_DIR / '05_categorical_analysis.png'}")
    print(f"   - {Config.OUTPUT_DIR / 'insights.md'}")
    print(f"   - data/processed/steam_processed.csv")
    print("\n📝 NEXT: Run Day 2 - Feature Engineering & Pipeline\n")


if __name__ == "__main__":
    main()