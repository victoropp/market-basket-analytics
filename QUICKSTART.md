# 🚀 Quick Start Guide

Get the Market Basket Analytics dashboard running in under 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- pip package manager
- 500MB free disk space

## Installation Steps

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/market-basket-analytics.git
cd market_basket_analytics
```

Or download and extract the ZIP file, then navigate to the folder.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including Streamlit, scikit-learn, mlxtend, and Plotly.

### 3. Launch the Dashboard

```bash
streamlit run deployment/app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## Using the Dashboard

### 📊 Executive Dashboard

View high-level KPIs and metrics:
- Total transactions, revenue, customers
- Geographic distribution
- Top-selling products
- Transaction trends over time

### 🔗 Association Rules Explorer

Discover product relationships:
- Filter rules by confidence, lift, and support
- Interactive scatter plots
- Sort and search through association rules
- Export rules to CSV

### 💡 Recommendation Engine

Get real-time product recommendations:
1. Select products from the dropdown (simulates customer basket)
2. Click "Get Recommendations"
3. View top product suggestions with confidence scores
4. See which associations drive each recommendation

**Example baskets to try:**
- "WHITE HANGING HEART T-LIGHT HOLDER" → High-lift decorative items
- "REGENCY CAKESTAND 3 TIER" → Kitchen and tea-time products
- "JUMBO BAG RED RETROSPOT" → Storage and organization items

### 🏢 Industry Use Cases

Explore 5 detailed industry applications:
- **Retail**: Product placement optimization
- **E-commerce**: Personalized cross-sell
- **Grocery**: Dynamic bundle pricing
- **Pharmacy**: Medication safety & complementary products
- **Restaurant**: Menu engineering & upselling

### 📈 Analytics & Insights

Deep dive into patterns:
- Temporal analysis (hourly, daily trends)
- Basket size and value distributions
- Customer purchase behavior
- Key business insights

## Advanced Usage

### Train Your Own Models

If you want to regenerate the analysis from scratch:

**Step 1: Preprocess data**
```bash
python src/data_preprocessing.py
```

**Step 2: Generate association rules**
```bash
python src/association_rules.py
```

**Step 3: Perform customer segmentation**
```bash
python src/segmentation.py
```

**Step 4: Launch dashboard**
```bash
streamlit run deployment/app.py
```

### Test Recommendations

Test the recommendation engine:
```bash
python scripts/test_recommendations.py
```

### Generate Social Media Assets

Create visualization assets for sharing:
```bash
python scripts/generate_social_media_plots.py
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Data files not found

**Solution**: The processed data files are included in the repository. If missing:
```bash
python src/data_preprocessing.py
python src/association_rules.py
python src/segmentation.py
```

### Issue: Port already in use

**Solution**: Specify a different port
```bash
streamlit run deployment/app.py --server.port 8502
```

### Issue: Slow dashboard loading

**Solution**:
- The dashboard loads pre-computed results for fast performance
- If regenerating rules, use higher min_support (0.01 instead of 0.005)
- Reduce max antecedent length in association rules

## Project Structure

```
market_basket_analytics/
├── deployment/app.py          # Main Streamlit dashboard
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning
│   ├── association_rules.py   # FP-Growth mining
│   ├── segmentation.py        # RFM & clustering
│   ├── recommendations.py     # Recommendation engine
│   └── utils.py               # Helper functions
├── scripts/                   # Utility scripts
│   ├── test_recommendations.py
│   ├── diagnose_products.py
│   └── generate_social_media_plots.py
├── models/                    # Saved models & rules
├── data/                      # Raw & processed data
└── requirements.txt           # Dependencies
```

## Next Steps

1. ✅ **Explore the Dashboard**: Try different products and filters
2. ✅ **View Association Rules**: Discover product relationships
3. ✅ **Test Recommendations**: See the engine in action
4. ✅ **Check Industry Use Cases**: Learn business applications
5. ✅ **Read Documentation**: Check README.md for detailed information

## Key Metrics to Explore

### Association Rules
- **Support**: How often items appear together (min: 0.5%)
- **Confidence**: Likelihood of consequent given antecedent (min: 30%)
- **Lift**: Strength of association vs random (min: 2.0x)

### Customer Segments
- **Champions**: High RFM scores - VIP treatment
- **Loyal Customers**: Regular buyers - upsell opportunities
- **Big Spenders**: High monetary - increase frequency
- **At Risk**: Declining activity - win-back campaigns

### Business Impact
- **20-35% increase** in AOV through cross-sell
- **15-25% boost** in basket size
- **30-40% improvement** in bundle sales

## Need Help?

- 📖 Check the main [README.md](README.md) for detailed documentation
- 🐛 Issues? Review troubleshooting section above
- 💡 Questions? Check the inline documentation in source code

---

**Ready to discover hidden patterns in your data!** 🎉

For advanced customization and API integration, refer to the main README.md file.
