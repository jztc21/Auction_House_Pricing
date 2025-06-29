# Auction_House_Pricing
Machine Learning Model using Random Forest Regression to predict hero copy prices on the auction house

This Python-based tool uses machine learning to predict the resale value of hero copies in the Idle Heroes Auction House. The model helps determine whether a specific listing is likely to turn at least a 5% profit.

## How It Works

- **Training Data**: Based on recent sales with full stats (`recent_sales.csv`) and historical sales without stats (`heroes.csv`).
- **Baseline Price**: Uses the average sale price of each hero at a given tier as the starting point.
- **Model Prediction**: The AI predicts a **price delta** above the baseline by analyzing stat differences (like attack, HP, speed).
- **Stat Comparison**: Adjusts predictions based on how the hero’s stats compare to the average for that hero and tier.
- **Final Prediction**: `Predicted Price = Baseline Price + Model Predicted Delta`

---

## Files

- `Tier_diff_included.py` – Main script to train the model and predict hero prices.
- `recent_sales.csv` – Historical data including hero name, tier, price, and stats.
- `heroes.csv` – Recent sales data with only hero name, tier, and price (used for baseline prices).
---

When you run the program, it will ask:
 - Hero Name
 - Hero Tier
 - Attack Stat
 - HP Stat
 - Speed Stat

