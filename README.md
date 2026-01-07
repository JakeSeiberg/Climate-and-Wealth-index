# Climate & Wealth Index Analysis

A comprehensive data analysis project investigating the relationship between national wealth and climate change contribution across 190+ countries using World Bank data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Key Findings](#key-findings)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project analyzes the correlation between national wealth and climate change contribution by creating two composite indexes from World Bank indicators:

1. **Climate Contribution Index**: Measures a country's contribution to climate change through emissions, fossil fuel consumption, and carbon-intensive activities
2. **General Wealth Index**: Measures overall national wealth and development through economic and social indicators

The analysis reveals a strong positive correlation (r > 0.7) between wealth and climate impact, demonstrating that wealthier nations contribute disproportionately to global climate change.

## Features

- **Automated Data Collection**: Fetches real-time data from World Bank API for 190+ countries
- **Composite Index Creation**: Combines multiple indicators using weighted scoring algorithms
- **Statistical Analysis**: Performs Pearson and Spearman correlation, ANOVA testing, and residual analysis
- **Outlier Detection**: Identifies countries that deviate significantly from the wealth-climate trend
- **Professional Visualizations**: Generates publication-quality charts and graphs
- **ArcGIS Integration**: Outputs CSV files compatible with GIS software for geospatial mapping
- **Comprehensive Reporting**: Creates detailed text reports with statistical summaries

## Methodology

### Climate Contribution Index

**Higher scores = Greater contribution to climate change**

The index focuses on per-capita metrics to ensure fair comparison across countries of different sizes:

- **CO2 Emissions per Capita** (40% weight): Primary indicator of individual impact
- **Energy Use per Capita** (20% weight): Overall energy consumption patterns
- **Electric Power Consumption per Capita** (10% weight): Electricity usage intensity
- **Fossil Fuel Energy Consumption** (15% weight): Dependence on carbon-intensive fuels
- **Fossil Fuel Electricity Generation** (10% weight): Clean vs. dirty electricity mix
- **Total CO2 Emissions** (5% weight): Absolute contribution to global emissions

### General Wealth Index

**Higher scores = Greater wealth and development**

Combines economic prosperity with social development indicators:

- **GNI per Capita PPP** (18% weight): Income adjusted for purchasing power
- **GDP per Capita PPP** (18% weight): Economic output per person
- **Life Expectancy** (15% weight): Health outcomes and healthcare quality
- **Literacy Rate** (12% weight): Educational attainment
- **Electricity Access** (10% weight): Infrastructure development
- **Tertiary Education Enrollment** (10% weight): Higher education access
- **Unemployment Rate** (8% weight, inverted): Labor market health
- **Maternal Mortality Ratio** (5% weight, inverted): Healthcare quality
- **Gini Index** (4% weight, inverted): Income inequality

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy wbgapi
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/JakeSeiberg/Climate-Wealth-Index-Analysis.git
cd Climate-Wealth-Index-Analysis
```

## Usage

### Step 1: Generate Index Data

Run the index calculator to fetch World Bank data and create the composite indexes:

```bash
python index_calculator.py
```

This will generate:
- `climate_contribution_index.csv` - Climate impact data for each country
- `wealth_index.csv` - Wealth and development data for each country

### Step 2: Perform Correlation Analysis

Run the correlation analysis to examine the relationship between wealth and climate impact:

```bash
python correlation_calculator.py
```

This will generate:
- `wealth_climate_correlation.png` - Statistical plots and visualizations
- `wealth_climate_detailed.png` - Detailed country-labeled scatter plot
- `correlation_analysis_report.txt` - Comprehensive text report

### Using the Data in ArcGIS

1. Add a world country shapefile to ArcGIS Pro or ArcMap
2. Right-click the shapefile → **Joins and Relates** → **Add Join**
3. Join on the **ISO3** country code field
4. Symbolize using `Climate_Contribution_Index` or `Wealth_Index` fields
5. Create choropleth maps showing global patterns

## Output Files

### CSV Files (ArcGIS Compatible)

| File | Description | Key Fields |
|------|-------------|------------|
| `climate_contribution_index.csv` | Climate impact by country | ISO3, Country_Name, Climate_Contribution_Index, Climate_Category, CO2_emissions_per_capita, Fossil_fuel_energy_consumption |
| `wealth_index.csv` | Wealth metrics by country | ISO3, Country_Name, Wealth_Index, Wealth_Category, GNI_per_capita_PPP, Life_expectancy |

### Visualizations

- **wealth_climate_correlation.png**: 4-panel statistical analysis
  - Scatter plot with regression line
  - Box plot by wealth category
  - Outlier identification plot
  - Residual plot for model diagnostics

- **wealth_climate_detailed.png**: Detailed country view
  - All countries plotted with trend line
  - Top contributors and outliers labeled
  - Color-coded by climate impact intensity

### Reports

- **correlation_analysis_report.txt**: Comprehensive statistical summary
  - Correlation coefficients and significance tests
  - Top/bottom countries by climate impact
  - Interpretation of findings

## Key Findings

### Statistical Results

- **Pearson Correlation**: r = 0.72 (p < 0.001)
- **Spearman Correlation**: ρ = 0.68 (p < 0.001)
- **R-squared**: 0.52 (52% of climate impact variance explained by wealth)

### Interpretation

1. **Strong Positive Correlation**: Wealthier nations consistently show higher per-capita climate impact
2. **Statistically Significant**: The relationship is highly significant (p < 0.001)
3. **Substantial Explanatory Power**: Over half of climate impact variation can be explained by wealth level
4. **Notable Outliers**: Some countries achieve high wealth with relatively low climate impact (e.g., Nordic countries with renewable energy), while others show high impact for their wealth level (e.g., oil-producing nations)

### Top Climate Contributors

Countries with highest Climate Contribution Index scores (2019 data):
- Qatar
- Kuwait
- United Arab Emirates
- Bahrain
- Trinidad and Tobago
- Saudi Arabia
- Australia
- United States
- Canada
- Luxembourg

### Policy Implications

- **Wealth enables higher consumption**: Economic development is strongly associated with increased energy use and emissions
- **Technology isn't enough**: Even wealthy nations with advanced technology show high per-capita impact
- **Decoupling is rare**: Few countries have achieved high wealth with low climate impact
- **Global responsibility**: Wealthier nations bear disproportionate responsibility for climate change

## Technologies

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization
- **SciPy**: Statistical analysis and hypothesis testing
- **wbgapi**: World Bank API access for indicator data
- **ArcGIS**: Geospatial visualization (external software)

## Project Structure

```
Climate-Wealth-Index-Analysis/
│
├── index_calculator.py              # Generate composite indexes from World Bank data
├── correlation_calculator.py        # Perform statistical correlation analysis
├── requirements.txt                 # Python package dependencies
├── README.md                        # Project documentation
│
├── climate_contribution_index.csv   # Output: Climate impact data (generated)
├── wealth_index.csv                 # Output: Wealth data (generated)
├── wealth_climate_correlation.png   # Output: Statistical plots (generated)
├── wealth_climate_detailed.png      # Output: Detailed visualization (generated)
└── correlation_analysis_report.txt  # Output: Statistical report (generated)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Jacob Seiberg**

- GitHub: [@JakeSeiberg](https://github.com/JakeSeiberg)
- LinkedIn: [Jacob Seiberg](https://www.linkedin.com/in/jacob-seiberg-55ba64335/)
- Portfolio: [pages.uoregon.edu/jseiberg](https://pages.uoregon.edu/jseiberg/index.html)

## Acknowledgments

- **World Bank**: For providing comprehensive open-access development indicators
- **University of Oregon**: For supporting research and data analysis education
- **Geographic Information Systems Community**: For inspiring the ArcGIS integration

## References

- World Bank Open Data: https://data.worldbank.org/
- World Bank API Documentation: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
- IPCC Climate Change Reports: https://www.ipcc.ch/
- Our World in Data (Climate): https://ourworldindata.org/co2-and-greenhouse-gas-emissions

---

**If you found this project useful, please consider giving it a star!**

**Questions or feedback?** Feel free to open an issue or contact me directly.