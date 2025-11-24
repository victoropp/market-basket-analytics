# 🛒 Market Basket Analysis: From Theory to Production

## About This Notebook
This project goes beyond standard "tutorial" market basket analysis. It demonstrates how to build a **robust, fault-tolerant recommendation engine** suitable for real-world retail environments.

## Key Features
1.  **Advanced Association Rule Mining:**
    - Comparison of Apriori vs. FP-Growth algorithms.
    - Optimization of `min_support` thresholds to balance computational cost vs. rule coverage.
    - **Result:** Generated **105,849 rules** (vs standard baseline of ~4,000).

2.  **Production-Grade Logic:**
    - **Partial Matching:** Most tutorials require a strict subset match. This engine uses set intersection to find relevant rules even if the user's basket doesn't perfectly match the antecedent.
    - **Smart Fallback:** What happens when a user selects a niche product with no rules? This system gracefully falls back to popularity-based recommendations, ensuring a 100% response rate.

3.  **Interactive Dashboard:**
    - A fully functional Streamlit app allowing users to simulate shopping baskets and see real-time cross-sell suggestions.

## Technical Highlights
# 🛒 Market Basket Analysis: From Theory to Production

## About This Notebook
This project goes beyond standard "tutorial" market basket analysis. It demonstrates how to build a **robust, fault-tolerant recommendation engine** suitable for real-world retail environments.

## Key Features
1.  **Advanced Association Rule Mining:**
    - Comparison of Apriori vs. FP-Growth algorithms.
    - Optimization of `min_support` thresholds to balance computational cost vs. rule coverage.
    - **Result:** Generated **105,849 rules** (vs standard baseline of ~4,000).

2.  **Production-Grade Logic:**
    - **Partial Matching:** Most tutorials require a strict subset match. This engine uses set intersection to find relevant rules even if the user's basket doesn't perfectly match the antecedent.
    - **Smart Fallback:** What happens when a user selects a niche product with no rules? This system gracefully falls back to popularity-based recommendations, ensuring a 100% response rate.

3.  **Interactive Dashboard:**
    - A fully functional Streamlit app allowing users to simulate shopping baskets and see real-time cross-sell suggestions.

## Technical Highlights
- **Libraries:** `mlxtend`, `pandas`, `streamlit`, `plotly`
- **Techniques:** Association Rules, Collaborative Filtering (Item-Item), Customer Segmentation (RFM)

## How to Use
1.  Run the preprocessing steps to clean the raw transaction data.
2.  Execute the FP-Growth algorithm (pre-tuned to 0.5% support).
3.  Launch the dashboard to explore the generated rules interactively.

*If you find this kernel helpful, please upvote!* 👍

## 📸 Visual Assets
- **Cover Image:** `social_media/kaggle_cover.png`
- **Analysis Plots:** `social_media/rules_scatter_plot.png`, `social_media/top_products_plot.png`
