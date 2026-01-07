
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the climate and wealth index CSV files"""
    print("Loading data files...")
    
    try:
        climate_df = pd.read_csv('climate_contribution_index.csv')
        wealth_df = pd.read_csv('wealth_index.csv')
        
        print(f"Loaded climate data: {len(climate_df)} countries")
        print(f"Loaded wealth data: {len(wealth_df)} countries")
        
        return climate_df, wealth_df
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find CSV files.")
        print(f"Please run the index_calculator.py script first to generate:")
        print(f"  - climate_contribution_index.csv")
        print(f"  - wealth_index.csv")
        raise


def merge_datasets(climate_df, wealth_df):
    """Merge climate and wealth data on ISO3 code"""
    print("\nMerging datasets...")
    
    merged_df = climate_df.merge(
        wealth_df[['ISO3', 'Wealth_Index', 'Wealth_Category']], 
        on='ISO3', 
        how='inner'
    )
    
    merged_df = merged_df.dropna(subset=['Climate_Contribution_Index', 'Wealth_Index'])
    
    print(f"✓ Merged dataset: {len(merged_df)} countries with both indexes")
    
    return merged_df


def calculate_correlations(merged_df):
    """Calculate various correlation metrics"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    climate = merged_df['Climate_Contribution_Index']
    wealth = merged_df['Wealth_Index']
    
    pearson_r, pearson_p = pearsonr(climate, wealth)
    spearman_r, spearman_p = spearmanr(climate, wealth)
    r_squared = pearson_r ** 2
    
    print(f"\n📊 Correlation Coefficients:")
    print(f"   Pearson's r:  {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    print(f"   Spearman's ρ: {spearman_r:.4f} (p-value: {spearman_p:.4e})")
    print(f"   R-squared:    {r_squared:.4f}")
    
    print(f"\n📈 Interpretation:")
    
    if pearson_p < 0.001:
        significance = "highly significant (p < 0.001)"
    elif pearson_p < 0.01:
        significance = "very significant (p < 0.01)"
    elif pearson_p < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    abs_r = abs(pearson_r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    direction = "positive" if pearson_r > 0 else "negative"
    
    print(f"   There is a {strength} {direction} correlation that is {significance}.")
    
    if pearson_r > 0:
        print(f"   → Wealthier nations tend to have HIGHER climate impact.")
    else:
        print(f"   → Wealthier nations tend to have LOWER climate impact.")
    
    print(f"\n   {r_squared*100:.1f}% of the variation in climate impact can be")
    print(f"   explained by a country's wealth level.")
    
    return pearson_r, spearman_r, r_squared, pearson_p


def analyze_by_category(merged_df):
    """Analyze climate impact across wealth categories"""
    print("\n" + "="*60)
    print("CLIMATE IMPACT BY WEALTH CATEGORY")
    print("="*60)
    
    category_stats = merged_df.groupby('Wealth_Category')['Climate_Contribution_Index'].agg([
        ('Count', 'count'),
        ('Mean_Climate_Impact', 'mean'),
        ('Median_Climate_Impact', 'median'),
        ('Std_Dev', 'std')
    ]).round(2)
    
    print("\n" + category_stats.to_string())
    
    wealth_categories = merged_df['Wealth_Category'].unique()
    groups = [merged_df[merged_df['Wealth_Category'] == cat]['Climate_Contribution_Index'].values 
              for cat in wealth_categories if cat in merged_df['Wealth_Category'].values]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\n📊 ANOVA Test:")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print(f"   → Climate impact differs SIGNIFICANTLY across wealth categories")
        else:
            print(f"   → Climate impact does NOT differ significantly across wealth categories")


def find_outliers(merged_df, pearson_r):
    """Identify countries that deviate significantly from the trend"""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)
    
    climate = merged_df['Climate_Contribution_Index']
    wealth = merged_df['Wealth_Index']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(wealth, climate)
    
    merged_df['Predicted_Climate'] = slope * merged_df['Wealth_Index'] + intercept
    merged_df['Residual'] = merged_df['Climate_Contribution_Index'] - merged_df['Predicted_Climate']
    merged_df['Abs_Residual'] = abs(merged_df['Residual'])
    
    residual_std = merged_df['Residual'].std()
    threshold = 1.5 * residual_std
    
    outliers = merged_df[merged_df['Abs_Residual'] > threshold].sort_values('Abs_Residual', ascending=False)
    
    if len(outliers) > 0:
        print("\n Countries that deviate significantly from the wealth-climate trend:\n")
        
        print("HIGH CLIMATE IMPACT for their wealth level (above the line):")
        high_outliers = outliers[outliers['Residual'] > 0].head(10)
        if len(high_outliers) > 0:
            for _, row in high_outliers.iterrows():
                print(f"   {row['Country_Name']:30s} (Wealth: {row['Wealth_Index']:.1f}, Climate: {row['Climate_Contribution_Index']:.1f})")
        else:
            print("   None found")
        
        print("\nLOW CLIMATE IMPACT for their wealth level (below the line):")
        low_outliers = outliers[outliers['Residual'] < 0].head(10)
        if len(low_outliers) > 0:
            for _, row in low_outliers.iterrows():
                print(f"   {row['Country_Name']:30s} (Wealth: {row['Wealth_Index']:.1f}, Climate: {row['Climate_Contribution_Index']:.1f})")
        else:
            print("   None found")
    
    return merged_df


def create_visualizations(merged_df, pearson_r, r_squared):
    """Create visualization plots"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    fig2, ax = plt.subplots(figsize=(14, 10))
    
    scatter = ax.scatter(merged_df['Wealth_Index'], 
                        merged_df['Climate_Contribution_Index'],
                        c=merged_df['Climate_Contribution_Index'],
                        cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(merged_df['Wealth_Index'], merged_df['Climate_Contribution_Index'], 1)
    p = np.poly1d(z)
    ax.plot(merged_df['Wealth_Index'], p(merged_df['Wealth_Index']), 
            "r--", linewidth=2, alpha=0.8, label=f'Trend line (r={pearson_r:.3f})')
    
    interesting = merged_df.nlargest(10, 'Climate_Contribution_Index').copy()
    interesting = pd.concat([interesting, merged_df.nsmallest(10, 'Climate_Contribution_Index')])
    interesting = pd.concat([interesting, merged_df.nlargest(5, 'Abs_Residual')])
    interesting = interesting.drop_duplicates(subset=['ISO3'])
    
    for _, row in interesting.iterrows():
        ax.annotate(row['ISO3'], 
                   xy=(row['Wealth_Index'], row['Climate_Contribution_Index']),
                   xytext=(3, 3), textcoords='offset points', 
                   fontsize=9, fontweight='bold', alpha=0.8)
    
    ax.set_xlabel('Wealth Index →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Climate Contribution Index →', fontsize=14, fontweight='bold')
    ax.set_title(f'Wealth vs Climate Impact by Country\nCorrelation: r = {pearson_r:.3f}, R² = {r_squared:.3f}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Climate Contribution Index', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('wealth_climate_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: wealth_climate_detailed.png")


def save_analysis_report(merged_df, pearson_r, spearman_r, r_squared, p_value):
    """Save a text report of the analysis"""
    with open('correlation_analysis_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("WEALTH VS CLIMATE IMPACT CORRELATION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {len(merged_df)} countries with complete data\n\n")
        
        f.write("CORRELATION STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pearson's r:        {pearson_r:.4f}\n")
        f.write(f"Spearman's:     {spearman_r:.4f}\n")
        f.write(f"R-squared:          {r_squared:.4f}\n")
        f.write(f"P-value:            {p_value:.4e}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        
        if abs(pearson_r) >= 0.7:
            strength = "STRONG"
        elif abs(pearson_r) >= 0.5:
            strength = "MODERATE"
        elif abs(pearson_r) >= 0.3:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"
        
        direction = "POSITIVE" if pearson_r > 0 else "NEGATIVE"
        
        f.write(f"There is a {strength} {direction} correlation between national wealth\n")
        f.write(f"and climate change contribution.\n\n")
        
        f.write(f"Approximately {r_squared*100:.1f}% of the variation in climate impact\n")
        f.write(f"can be explained by a country's wealth level.\n\n")
        
        if p_value < 0.001:
            f.write("This correlation is HIGHLY STATISTICALLY SIGNIFICANT (p < 0.001).\n\n")
        elif p_value < 0.05:
            f.write("This correlation is STATISTICALLY SIGNIFICANT (p < 0.05).\n\n")
        else:
            f.write("This correlation is NOT statistically significant (p >= 0.05).\n\n")
    
    print("✓ Saved report: correlation_analysis_report.txt")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("WEALTH VS CLIMATE IMPACT CORRELATION ANALYSIS")
    print("="*70)
    
    try:
        # Load data
        climate_df, wealth_df = load_data()
        
        # Merge datasets
        merged_df = merge_datasets(climate_df, wealth_df)
        
        # Calculate correlations
        pearson_r, spearman_r, r_squared, p_value = calculate_correlations(merged_df)
        
        # Analyze by wealth category
        analyze_by_category(merged_df)
        
        # Find outliers
        merged_df = find_outliers(merged_df, pearson_r)
        
        # Create visualizations
        create_visualizations(merged_df, pearson_r, r_squared)
        
        # Save report
        save_analysis_report(merged_df, pearson_r, spearman_r, r_squared, p_value)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  wealth_climate_detailed.png - Detailed country view")
        print("  correlation_analysis_report.txt - Full text report")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()