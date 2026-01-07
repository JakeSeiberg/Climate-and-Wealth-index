
import pandas as pd
import numpy as np
import wbgapi as wb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IndexCalculator:
    """Calculate composite indexes from World Bank indicators"""
    
    def __init__(self, year: int = 2019):
        """
        Initialize calculator
        
        Args:
            year: Year for data collection (default: 2019, best emissions data coverage)
        """
        self.year = year
        
    def fetch_data(self, indicators: Dict[str, str], db: int = 2) -> pd.DataFrame:
        """
        Fetch data from World Bank API
        
        Args:
            indicators: Dictionary mapping indicator codes to descriptive names
            db: Database ID (2=WDI, 75=Climate Change, etc.)
            
        Returns:
            DataFrame with countries and indicator values (excludes aggregates/regions)
        """
        print(f"Fetching data for year {self.year} from database {db}...")
        
        data_frames = []
        for code, name in indicators.items():
            try:
                print(f"  Fetching {name}...")
                df = wb.data.DataFrame(code, time=self.year, db=db, skipBlanks=True, columns='series')
                df = df.reset_index()
                df.columns = ['Country', name]
                data_frames.append(df)
            except Exception as e:
                print(f"  Warning: Could not fetch {name}: {e}")
                
        if not data_frames:
            raise ValueError("No data could be fetched")
            
        result = data_frames[0]
        for df in data_frames[1:]:
            result = result.merge(df, on='Country', how='outer')
        
        print("  Filtering to include only countries (excluding regional aggregates)...")
        result = self.filter_countries_only(result)
            
        return result
    
    def filter_countries_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to include only actual countries (sovereign states), 
        excluding territories, dependencies, and regional aggregates.
        Includes Kosovo even though it may not have full World Bank classification.
        
        Args:
            df: DataFrame with Country column containing ISO3 codes
            
        Returns:
            DataFrame with only countries (no aggregates, no territories)
        """
        try:
            economies = wb.economy.DataFrame()
            economies = economies.reset_index()
            
            valid_lending = ['IBD', 'IDB', 'IDX', 'LNX']  # Include "Not classified" for special cases
            
            if 'lendingType' in economies.columns and 'region' in economies.columns:
                country_codes = economies[
                    (economies['lendingType'].isin(valid_lending)) & 
                    (economies['region'].notna()) &
                    (economies['region'] != '')
                ]['id'].tolist()
            elif 'lendingType' in economies.columns:
                country_codes = economies[economies['lendingType'].isin(valid_lending)]['id'].tolist()
            else:
                country_codes = economies[
                    (economies['region'].notna()) & 
                    (economies['region'] != '')
                ]['id'].tolist()
            
            if 'XKX' not in country_codes:
                country_codes.append('XKX')
            
            territories_to_exclude = [
                'ABW', 'ASM', 'BMU', 'CUW', 'CYM', 'FRO', 'GIB', 'GRL', 
                'GUM', 'HKG', 'IMN', 'MAC', 'MNP', 'NCL', 'PRI', 'PYF',
                'SXM', 'TCA', 'VGB', 'VIR', 'WLF', 'CHI' 
            ]
            
            country_codes = [c for c in country_codes if c not in territories_to_exclude]
            
            df_filtered = df[df['Country'].isin(country_codes)].copy()
            
            print(f"  Filtered from {len(df)} to {len(df_filtered)} entities (countries only, territories excluded)")
            
            return df_filtered
            
        except Exception as e:
            print(f"  Warning: Could not filter with lending type: {e}")
            print("  Attempting manual filtering of common aggregates and territories...")
            
            exclude_codes = [
                'WLD', 'EAS', 'ECS', 'LCN', 'MEA', 'NAC', 'SAS', 'SSF',
                'ARB', 'CSS', 'EUU', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA',
                'IDB', 'IDX', 'INX', 'LAC', 'LDC', 'LIC', 'LMC', 'LMY',
                'LTE', 'MIC', 'MNA', 'OED', 'OSS', 'PRE', 'PSS', 'PST',
                'SSA', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS',
                'UMC', 'FCS', 'EMU', 'EAP', 'ECA', 'LCR', 'MENA', 'SSD',
                'ABW', 'ASM', 'BMU', 'CUW', 'CYM', 'FRO', 'GIB', 'GRL',
                'GUM', 'HKG', 'IMN', 'MAC', 'MNP', 'NCL', 'PRI', 'PYF',
                'SXM', 'TCA', 'VGB', 'VIR', 'WLF', 'CHI'
            ]
            
            df_filtered = df[~df['Country'].isin(exclude_codes)].copy()
            
            if 'XKX' in df['Country'].values and 'XKX' not in df_filtered['Country'].values:
                kosovo_row = df[df['Country'] == 'XKX']
                df_filtered = pd.concat([df_filtered, kosovo_row], ignore_index=True)
            
            print(f"  Filtered from {len(df)} to {len(df_filtered)} entities")
            
            return df_filtered
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str], 
                       higher_is_worse: List[str] = None) -> pd.DataFrame:
        """
        Normalize data to 0-100 scale
        
        Args:
            df: Input dataframe
            columns: Columns to normalize
            higher_is_worse: Columns where higher values = worse impact (normal scoring)
            
        Returns:
            DataFrame with normalized columns
        """
        higher_is_worse = higher_is_worse or []
        df_norm = df.copy()
        
        for col in columns:
            if col in df_norm.columns:
                values = df_norm[col].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    
                    if max_val - min_val > 0:
                        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val) * 100
                            
        return df_norm
    
    def calculate_index(self, df: pd.DataFrame, columns: List[str], 
                       weights: Dict[str, float] = None) -> pd.Series:
        """
        Calculate composite index from normalized indicators
        
        Args:
            df: DataFrame with normalized indicators
            columns: Columns to include in index
            weights: Optional dictionary of weights for each column (must sum to 1)
            
        Returns:
            Series with index values
        """
        available_cols = [col for col in columns if col in df.columns]
        
        if not available_cols:
            raise ValueError("No valid columns found for index calculation")
        
        if weights is None:
            weights = {col: 1.0/len(available_cols) for col in available_cols}
        else:
            total = sum(weights.get(col, 0) for col in available_cols)
            weights = {col: weights.get(col, 0)/total for col in available_cols}
        
        index = pd.Series(0.0, index=df.index)
        total_weights = pd.Series(0.0, index=df.index)
        
        for col in available_cols:
            mask = df[col].notna()
            index[mask] += df[col][mask] * weights[col]
            total_weights[mask] += weights[col]
        
        index = index / total_weights
        index[total_weights == 0] = np.nan
        
        return index
    
    def add_iso_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ISO3 country codes for ArcGIS joining
        
        The World Bank API returns ISO3 codes as the country identifier.
        We need to extract them from the 'Country' column which contains the codes.
        
        Args:
            df: DataFrame with Country column (contains ISO3 codes)
            
        Returns:
            DataFrame with ISO3 codes and country names
        """
        try:
            df['ISO3'] = df['Country'].copy()
            
            countries = wb.economy.DataFrame()
            countries = countries.reset_index()
            
            country_map = dict(zip(countries['id'], countries['name']))
            
            df['Country_Name'] = df['ISO3'].map(country_map)
            
            other_cols = [col for col in df.columns if col not in ['ISO3', 'Country_Name', 'Country']]
            df = df[['ISO3', 'Country_Name'] + other_cols]
            
            if 'Country' in df.columns:
                df = df.drop('Country', axis=1)
            
        except Exception as e:
            print(f"Warning: Could not add ISO codes properly: {e}")
            df['ISO3'] = df['Country']
            df = df.drop('Country', axis=1)
            cols = ['ISO3'] + [col for col in df.columns if col != 'ISO3']
            df = df[cols]
            
        return df
    
    def save_for_arcgis(self, df: pd.DataFrame, filename: str):
        """
        Save data in ArcGIS-compatible CSV format
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        df = df.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(2)
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nSaved to {filename}")
        print(f"Records: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")


def create_climate_contribution_index(year: int = 2019) -> pd.DataFrame:
    """
    Create Climate Change Contribution Index
    
    HIGHER SCORES = GREATER CONTRIBUTION TO CLIMATE CHANGE (WORSE)
    
    This index measures a country's contribution to climate change through:
    - Greenhouse gas emissions (total and per capita)
    - Fossil fuel consumption and production
    - Carbon-intensive economic activities
    - Deforestation and land use changes
    
    Countries like Saudi Arabia, Qatar, UAE, USA should score high
    """
    print("\n" + "="*60)
    print("CREATING CLIMATE CHANGE CONTRIBUTION INDEX")
    print("="*60)
    print("Methodology: Higher scores = Greater contribution to climate change")
    
    calculator = IndexCalculator(year)
        
    print("\n--- Fetching EMISSIONS data (per capita focus) ---")
    emissions_data = {}
    
    try:
        print("  Attempting Climate Change database (source 75)...")
        df_co2_pc = wb.data.DataFrame('EN.ATM.CO2E.PC', time=year, db=75, skipBlanks=True, columns='series')
        df_co2_pc = df_co2_pc.reset_index()
        df_co2_pc.columns = ['Country', 'CO2_emissions_per_capita']
        emissions_data['CO2_emissions_per_capita'] = df_co2_pc
        print("  CO2 per capita fetched successfully")
    except Exception as e:
        print(f"  Warning: Could not fetch CO2 per capita: {e}")
    
    try:
        print("  Attempting WDI database for total emissions...")
        df_co2_total = wb.data.DataFrame('EN.ATM.CO2E.KT', time=year, db=2, skipBlanks=True, columns='series')
        df_co2_total = df_co2_total.reset_index()
        df_co2_total.columns = ['Country', 'CO2_emissions_total_kt']
        emissions_data['CO2_emissions_total'] = df_co2_total
        print("  Total CO2 emissions fetched successfully")
    except Exception as e:
        print(f"  Warning: Could not fetch total CO2: {e}")
    
    energy_indicators = {
        'EG.USE.COMM.FO.ZS': 'Fossil_fuel_energy_consumption',
        'EG.ELC.FOSL.ZS': 'Fossil_fuel_electricity',
        'EG.USE.PCAP.KG.OE': 'Energy_use_per_capita',
        'EG.USE.ELEC.KH.PC': 'Electric_power_consumption_per_capita'
    }
    
    print("\n--- Fetching ENERGY data from World Development Indicators ---")
    df_energy = calculator.fetch_data(energy_indicators, db=2)
    
    df = df_energy.copy()
    
    for name, em_df in emissions_data.items():
        print(f"  Merging {name}...")
        em_df_filtered = calculator.filter_countries_only(em_df)
        df = df.merge(em_df_filtered, on='Country', how='outer')
    
    print(f"\nFinal dataset has {len(df)} countries with {len(df.columns)-1} indicators")
    
    index_cols = []
    available_indicators = df.columns.tolist()
    available_indicators.remove('Country')  # Don't include Country in index
    
    # Add indicators if they exist
    potential_cols = [
        'CO2_emissions_per_capita',
        'CO2_emissions_total_kt',
        'Fossil_fuel_energy_consumption',
        'Fossil_fuel_electricity',
        'Energy_use_per_capita',
        'Electric_power_consumption_per_capita'
    ]
    
    for col in potential_cols:
        if col in available_indicators:
            index_cols.append(col)
            print(f"  Including: {col}")
    
    if not index_cols:
        raise ValueError("No valid indicators available for index calculation!")
    
    print(f"\nBuilding index from {len(index_cols)} indicators")
    
    print("\nNormalizing indicators (higher = worse for climate)...")
    df_norm = calculator.normalize_data(df, index_cols, higher_is_worse=index_cols)
    
    weights = {}
    
    if 'CO2_emissions_per_capita' in index_cols:
        weights['CO2_emissions_per_capita'] = 0.40 
    if 'Energy_use_per_capita' in index_cols:
        weights['Energy_use_per_capita'] = 0.20 
    if 'Electric_power_consumption_per_capita' in index_cols:
        weights['Electric_power_consumption_per_capita'] = 0.10
    if 'Fossil_fuel_energy_consumption' in index_cols:
        weights['Fossil_fuel_energy_consumption'] = 0.15
    if 'Fossil_fuel_electricity' in index_cols:
        weights['Fossil_fuel_electricity'] = 0.10
    if 'CO2_emissions_total_kt' in index_cols:
        weights['CO2_emissions_total_kt'] = 0.05
    
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    print("\nWeights used:")
    for k, v in weights.items():
        print(f"  {k}: {v:.2%}")
    
    print("Calculating Climate Change Contribution Index...")
    df['Climate_Contribution_Index'] = calculator.calculate_index(df_norm, index_cols, weights)
    
    df = calculator.add_iso_codes(df)
    
    df['Climate_Category'] = pd.cut(df['Climate_Contribution_Index'], 
                                     bins=[0, 25, 50, 75, 100],
                                     labels=['Low Contributor', 'Moderate Contributor', 
                                            'High Contributor', 'Very High Contributor'])
    
    return df


def create_wealth_index(year: int = 2019) -> pd.DataFrame:
    """
    Create General Wealth Index
    
    Higher scores indicate greater wealth/development
    """
    print("\n" + "="*60)
    print("CREATING GENERAL WEALTH INDEX")
    print("="*60)
    
    calculator = IndexCalculator(year)
    
    indicators = {
        'NY.GNP.PCAP.PP.KD': 'GNI_per_capita_PPP',
        'NY.GDP.PCAP.PP.KD': 'GDP_per_capita_PPP',
        'SP.DYN.LE00.IN': 'Life_expectancy',
        'SE.ADT.LITR.ZS': 'Literacy_rate',
        'EG.ELC.ACCS.ZS': 'Electricity_access',
        'SE.TER.ENRR': 'Tertiary_enrollment',
        'SL.UEM.TOTL.ZS': 'Unemployment_rate',
        'SH.STA.MMRT': 'Maternal_mortality_ratio',
        'SI.POV.GINI': 'Gini_index'
    }
    
    df = calculator.fetch_data(indicators)
    
    positive_cols = [
        'GNI_per_capita_PPP',
        'GDP_per_capita_PPP',
        'Life_expectancy',
        'Literacy_rate',
        'Electricity_access',
        'Tertiary_enrollment'
    ]
    
    negative_cols = [
        'Unemployment_rate',
        'Maternal_mortality_ratio',
        'Gini_index'
    ]
    
    print("\nNormalizing indicators...")
    df_norm = calculator.normalize_data(df, positive_cols)
    
    for col in negative_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                if max_val - min_val > 0:
                    df_norm[col] = 100 - ((df[col] - min_val) / (max_val - min_val) * 100)
    
    all_cols = positive_cols + negative_cols
    
    weights = {
        'GNI_per_capita_PPP': 0.18,
        'GDP_per_capita_PPP': 0.18,
        'Life_expectancy': 0.15,
        'Literacy_rate': 0.12,
        'Electricity_access': 0.10,
        'Tertiary_enrollment': 0.10,
        'Unemployment_rate': 0.08,
        'Maternal_mortality_ratio': 0.05,
        'Gini_index': 0.04
    }
    
    print("Calculating General Wealth Index...")
    df['Wealth_Index'] = calculator.calculate_index(df_norm, all_cols, weights)
    
    df = calculator.add_iso_codes(df)
    
    df['Wealth_Category'] = pd.cut(df['Wealth_Index'], 
                                    bins=[0, 25, 50, 75, 100],
                                    labels=['Low', 'Lower-Middle', 
                                           'Upper-Middle', 'High'])
    
    return df


def main():
    """Main execution function"""
    
    print("\nWorld Bank Index Calculator for ArcGIS")
    print("=" * 60)
    
    year = 2019
    
    try:
        climate_df = create_climate_contribution_index(year)
        calculator = IndexCalculator(year)
        calculator.save_for_arcgis(climate_df, 'climate_contribution_index.csv')
        
        print("\n" + "="*60)
        print("CLIMATE CONTRIBUTION INDEX RESULTS")
        print("="*60)
        print("\nIndex Summary Statistics:")
        print(climate_df['Climate_Contribution_Index'].describe())
        
        wealth_df = create_wealth_index(year)
        calculator.save_for_arcgis(wealth_df, 'wealth_index.csv')
        
        print("\n" + "="*60)
        print("WEALTH INDEX RESULTS")
        print("="*60)
        print("\nIndex Summary Statistics:")
        print(wealth_df['Wealth_Index'].describe())
        
        print("\n" + "="*60)
        print("SUCCESS! Files created:")
        print("  - climate_contribution_index.csv")
        print("  - wealth_index.csv")
        
        print("\n Index Interpretation:")
        print("  Climate Index: Higher score = Greater contribution to climate change")
        print("  Wealth Index: Higher score = Greater wealth/development")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()