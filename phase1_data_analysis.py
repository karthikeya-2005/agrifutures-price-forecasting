"""Phase 1: Comprehensive Data Analysis & Exploration"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

class DataAnalyzer:
    def __init__(self, data_path):
        """Initialize with path to dataset"""
        self.data_path = Path(data_path)
        self.df = None
        self.summary = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("="*80)
        print("PHASE 1: DATA ANALYSIS & EXPLORATION")
        print("="*80 + "\n")
        
        print(f"Loading dataset: {self.data_path}")
        self.df = pd.read_csv(self.data_path, parse_dates=['date'], low_memory=False)
        print(f"[SUCCESS] Loaded {len(self.df):,} records\n")
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistics"""
        print("="*80)
        print("BASIC STATISTICS")
        print("="*80 + "\n")
        
        stats = {
            'total_records': len(self.df),
            'date_range': {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max()),
                'days': (self.df['date'].max() - self.df['date'].min()).days
            },
            'geographic': {
                'states': int(self.df['state'].nunique()),
                'districts': int(self.df['district'].nunique())
            },
            'commodities': int(self.df['crop'].nunique()),
            'columns': list(self.df.columns)
        }
        
        print(f"Total Records: {stats['total_records']:,}")
        print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"States: {stats['geographic']['states']}")
        print(f"Districts: {stats['geographic']['districts']}")
        print(f"Commodities: {stats['commodities']}")
        
        self.summary['basic_stats'] = stats
        return stats
    
    def data_quality_assessment(self):
        """Assess data quality"""
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80 + "\n")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        print("Missing Values:")
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("  [OK] No missing values!")
        
        # Price statistics
        print("\nPrice Statistics:")
        if 'price' in self.df.columns:
            price_stats = self.df['price'].describe()
            print(f"  Count:    {price_stats['count']:,.0f}")
            print(f"  Mean:     Rs. {price_stats['mean']:,.2f}")
            print(f"  Median:   Rs. {price_stats['50%']:,.2f}")
            print(f"  Std Dev:  Rs. {price_stats['std']:,.2f}")
            print(f"  Min:      Rs. {price_stats['min']:,.2f}")
            print(f"  Max:      Rs. {price_stats['max']:,.2f}")
            
            # Outliers
            Q1 = self.df['price'].quantile(0.25)
            Q3 = self.df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df['price'] < lower_bound) | (self.df['price'] > upper_bound)]
            print(f"\n  Outliers (IQR method): {len(outliers):,} ({len(outliers)/len(self.df)*100:.2f}%)")
            
            quality = {
                'missing_values': missing_df.to_dict('records') if len(missing_df) > 0 else [],
                'price_stats': price_stats.to_dict(),
                'outliers_count': int(len(outliers)),
                'outliers_pct': float(len(outliers)/len(self.df)*100)
            }
        else:
            quality = {'missing_values': missing_df.to_dict('records') if len(missing_df) > 0 else []}
        
        self.summary['data_quality'] = quality
        return quality
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS")
        print("="*80 + "\n")
        
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        
        print("Records by Year:")
        year_counts = self.df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count:,} records")
        
        print("\nRecords by Month (across all years):")
        month_counts = self.df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in month_counts.items():
            print(f"  {month_names[month-1]}: {count:,} records")
        
        temporal = {
            'year_distribution': year_counts.to_dict(),
            'month_distribution': month_counts.to_dict()
        }
        
        self.summary['temporal'] = temporal
        return temporal
    
    def geographic_analysis(self):
        """Analyze geographic patterns"""
        print("\n" + "="*80)
        print("GEOGRAPHIC ANALYSIS")
        print("="*80 + "\n")
        
        print("Top 10 States by Records:")
        state_counts = self.df['state'].value_counts().head(10)
        for state, count in state_counts.items():
            print(f"  {state}: {count:,} records")
        
        print("\nTop 10 Districts by Records:")
        district_counts = self.df['district'].value_counts().head(10)
        for district, count in district_counts.items():
            print(f"  {district}: {count:,} records")
        
        geographic = {
            'top_states': state_counts.to_dict(),
            'top_districts': district_counts.to_dict()
        }
        
        self.summary['geographic'] = geographic
        return geographic
    
    def commodity_analysis(self):
        """Analyze commodity patterns"""
        print("\n" + "="*80)
        print("COMMODITY ANALYSIS")
        print("="*80 + "\n")
        
        print("Top 20 Commodities by Records:")
        crop_counts = self.df['crop'].value_counts().head(20)
        for crop, count in crop_counts.items():
            print(f"  {crop}: {count:,} records")
        
        # Price statistics by commodity
        print("\nPrice Statistics by Top Commodities:")
        top_crops = crop_counts.head(10).index
        crop_price_stats = {}
        for crop in top_crops:
            crop_data = self.df[self.df['crop'] == crop]['price']
            crop_price_stats[crop] = {
                'mean': float(crop_data.mean()),
                'median': float(crop_data.median()),
                'std': float(crop_data.std()),
                'count': int(len(crop_data))
            }
            print(f"  {crop}: Mean=Rs. {crop_data.mean():,.2f}, Median=Rs. {crop_data.median():,.2f}")
        
        commodity = {
            'top_commodities': crop_counts.to_dict(),
            'price_by_commodity': crop_price_stats
        }
        
        self.summary['commodity'] = commodity
        return commodity
    
    def pattern_identification(self):
        """Identify patterns in data"""
        print("\n" + "="*80)
        print("PATTERN IDENTIFICATION")
        print("="*80 + "\n")
        
        patterns = {}
        
        # Seasonal patterns
        print("Seasonal Price Patterns (by Month):")
        monthly_avg_price = self.df.groupby('month')['price'].mean()
        print(monthly_avg_price.to_string())
        patterns['seasonal'] = monthly_avg_price.to_dict()
        
        # Day of week patterns
        print("\nDay of Week Price Patterns:")
        dow_avg_price = self.df.groupby('day_of_week')['price'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for dow, price in dow_avg_price.items():
            print(f"  {dow_names[dow]}: Rs. {price:,.2f}")
        patterns['day_of_week'] = dow_avg_price.to_dict()
        
        # Geographic price variations
        print("\nGeographic Price Variations (Top 10 States):")
        state_avg_price = self.df.groupby('state')['price'].mean().sort_values(ascending=False).head(10)
        for state, price in state_avg_price.items():
            print(f"  {state}: Rs. {price:,.2f}")
        patterns['geographic_price'] = state_avg_price.to_dict()
        
        self.summary['patterns'] = patterns
        return patterns
    
    def generate_visualizations(self, output_dir='data/analysis'):
        """Generate visualization plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Price distribution
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['price'], bins=100, edgecolor='black')
        plt.xlabel('Price (Rs.)')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_path / 'price_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] Saved: price_distribution.png")
        
        # Temporal trends
        monthly_prices = self.df.groupby(['year', 'month'])['price'].mean().reset_index()
        monthly_prices['date'] = pd.to_datetime(monthly_prices[['year', 'month']].assign(day=1))
        monthly_prices = monthly_prices.sort_values('date')
        
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_prices['date'], monthly_prices['price'], linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Average Price (Rs.)')
        plt.title('Average Price Trend Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'price_trend.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] Saved: price_trend.png")
        
        # Top commodities
        top_crops = self.df['crop'].value_counts().head(15)
        plt.figure(figsize=(12, 8))
        top_crops.plot(kind='barh')
        plt.xlabel('Number of Records')
        plt.ylabel('Commodity')
        plt.title('Top 15 Commodities by Record Count')
        plt.tight_layout()
        plt.savefig(output_path / 'top_commodities.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] Saved: top_commodities.png")
        
        # Geographic distribution
        top_states = self.df['state'].value_counts().head(15)
        plt.figure(figsize=(12, 8))
        top_states.plot(kind='barh')
        plt.xlabel('Number of Records')
        plt.ylabel('State')
        plt.title('Top 15 States by Record Count')
        plt.tight_layout()
        plt.savefig(output_path / 'top_states.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] Saved: top_states.png")
        
        print(f"\n[SUCCESS] All visualizations saved to: {output_path}")
    
    def save_summary(self, output_file='data/analysis/data_analysis_summary.json'):
        """Save analysis summary"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.summary['analysis_date'] = datetime.now().isoformat()
        self.summary['data_file'] = str(self.data_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.summary, f, indent=2, default=str)
        
        print(f"\n[SUCCESS] Analysis summary saved to: {output_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.load_data()
        self.basic_statistics()
        self.data_quality_assessment()
        self.temporal_analysis()
        self.geographic_analysis()
        self.commodity_analysis()
        self.pattern_identification()
        self.generate_visualizations()
        self.save_summary()
        
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE: DATA ANALYSIS & EXPLORATION")
        print("="*80)
        print("\nNext: Phase 2 - Feature Engineering")
        return self.summary

if __name__ == "__main__":
    data_file = Path('data/kaggle_combined/all_kaggle_final_complete.csv')
    
    if not data_file.exists():
        print(f"[ERROR] Dataset not found: {data_file}")
        exit(1)
    
    analyzer = DataAnalyzer(data_file)
    summary = analyzer.run_full_analysis()

