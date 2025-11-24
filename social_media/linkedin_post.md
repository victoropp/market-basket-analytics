# 🚀 LinkedIn Post Draft

**Headline:** How I Optimized a Recommendation Engine by 25x (and why "Support" matters)

I just wrapped up my latest project: a production-ready **Market Basket Analysis Dashboard** built with Python and Streamlit. 

The goal? Build a system that doesn't just find obvious patterns (like "Coffee + Sugar"), but uncovers the hidden gems that drive real cross-sell revenue.

**The Challenge:**
My initial model using the Apriori algorithm was "safe." It found ~4,000 strong association rules. Good, but not great. It left 93% of the product catalog without specific recommendations.

**The Optimization:**
I switched to the **FP-Growth** algorithm and fine-tuned the support thresholds. 
- 0.1% support was too computationally expensive (50+ min training).
- 1.0% support was too sparse (missed niche items).
- **0.5% support was the sweet spot.**

**The Result:**
- 📈 **Rules Generated:** Increased from 4,072 → **105,849** (25x improvement!)
- 🎯 **Coverage:** Recommendations now available for the vast majority of the catalog.
- 🛡️ **Reliability:** Implemented a "Smart Fallback" system using popularity metrics for cold-start items.

**Tech Stack:**
- **Core:** Python, Pandas, MLxtend
- **Viz:** Plotly, Streamlit
- **Architecture:** Modular design with separate data/model layers

Check out the full code and live demo below! 👇

[Link to GitHub Repo]
[Link to Streamlit App]

**📸 Assets to Attach:**
1.  `social_media/linkedin_cover.png` (Cover Image)
2.  `social_media/rules_scatter_plot.png` (Technical Proof)
3.  `social_media/top_products_plot.png` (Data Insight)

#DataScience #MachineLearning #Python #Streamlit #Portfolio #MarketBasketAnalysis #Optimization
