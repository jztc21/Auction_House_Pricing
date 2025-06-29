import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Define valid tiers and map them to numbers
valid_tiers = ['B+', 'A-', 'A', 'A+']
tier_map = {'B+': 1, 'A-': 2, 'A': 3, 'A+': 4}

# Load recent sales with stats
df = pd.read_csv('recent_sales.csv')
df = df[df['tier'].isin(valid_tiers)]
df['tier_score'] = df['tier'].map(tier_map)

# Load last 5 sales data (no stats)
last5 = pd.read_csv('heroes.csv')
last5 = last5[last5['tier'].isin(valid_tiers)]
last5['tier_score'] = last5['tier'].map(tier_map)

# Calculate baseline prices: average resale price per hero-tier
hero_tier_avg_price = last5.groupby(['hero_name', 'tier_score'])['price'].mean().to_dict()

# Calculate average stats per hero-tier from recent sales (for stat differences)
hero_tier_atk_avg = df.groupby(['hero_name', 'tier'])['atk'].mean().to_dict()
hero_tier_hp_avg = df.groupby(['hero_name', 'tier'])['hp'].mean().to_dict()
hero_tier_spd_avg = df.groupby(['hero_name', 'tier'])['spd'].mean().to_dict()

# Compute stat_score (weighted sum)
df['stat_score'] = df['atk'] * 0.85 + df['hp'] * 0.1 + df['spd'] * 0.05

# Calculate absolute stat differences (stat - avg stat for hero-tier)
def stat_diff(stat, hero_name, tier, avg_dict):
    avg = avg_dict.get((hero_name, tier))
    return stat - avg if avg is not None else 0

df['atk_diff'] = df.apply(lambda r: stat_diff(r['atk'], r['hero_name'], r['tier'], hero_tier_atk_avg), axis=1)
df['hp_diff'] = df.apply(lambda r: stat_diff(r['hp'], r['hero_name'], r['tier'], hero_tier_hp_avg), axis=1)
df['spd_diff'] = df.apply(lambda r: stat_diff(r['spd'], r['hero_name'], r['tier'], hero_tier_spd_avg), axis=1)

# Assign stat grades
def stat_grade(val):
    if val >= 8700: return 'SS'
    elif val >= 8100: return 'S'
    elif val >= 7200: return 'A+'
    elif val >= 6300: return 'A'
    elif val >= 5400: return 'A-'
    elif val >= 4400: return 'B+'
    else: return 'B'

for stat_col in ['atk', 'hp', 'spd']:
    df[f'{stat_col}_grade'] = df[stat_col].apply(stat_grade)

# Encode stat grades
grade_levels = ['B', 'B+', 'A-', 'A', 'A+', 'SS', 'S']
grade_encoders = {}
for col in ['atk_grade', 'hp_grade', 'spd_grade']:
    le = LabelEncoder()
    le.fit(grade_levels)
    df[col] = le.transform(df[col])
    grade_encoders[col] = le

# Encode hero names
hero_encoder = LabelEncoder()
df['hero_id'] = hero_encoder.fit_transform(df['hero_name'])

# Compute target: price delta = actual resale_price - baseline price for hero-tier
def get_baseline_price(row):
    return hero_tier_avg_price.get((row['hero_name'], row['tier_score']), df['resale_price'].median())

df['baseline_price'] = df.apply(get_baseline_price, axis=1)
df['price_delta'] = df['resale_price'] - df['baseline_price']

# Features for model — use tier_score, stat diffs, stat grades, hero_id, etc.
features = [
    'tier_score',
    'atk_diff', 'hp_diff', 'spd_diff',
    'atk_grade', 'hp_grade', 'spd_grade',
    'hero_id',
]
target = 'price_delta'

# Train/test split and train model
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model trained. MAE on delta: {mean_absolute_error(y_test, model.predict(X_test)):.2f} starry gems")

# Predict resale price: baseline + predicted delta
def predict_resale_price(hero_name, tier, atk, hp, spd):
    if tier not in tier_map:
        raise ValueError(f"Invalid tier '{tier}'")

    tier_score = tier_map[tier]

    # Calculate stat diffs relative to hero-tier average
    atk_d = stat_diff(atk, hero_name, tier, hero_tier_atk_avg)
    hp_d = stat_diff(hp, hero_name, tier, hero_tier_hp_avg)
    spd_d = stat_diff(spd, hero_name, tier, hero_tier_spd_avg)

    atk_g = grade_encoders['atk_grade'].transform([stat_grade(atk)])[0]
    hp_g = grade_encoders['hp_grade'].transform([stat_grade(hp)])[0]
    spd_g = grade_encoders['spd_grade'].transform([stat_grade(spd)])[0]

    if hero_name not in hero_encoder.classes_:
        raise ValueError(f"Unknown hero '{hero_name}'")
    hero_id = hero_encoder.transform([hero_name])[0]

    feature_row = pd.DataFrame([{
        'tier_score': tier_score,
        'atk_diff': atk_d,
        'hp_diff': hp_d,
        'spd_diff': spd_d,
        'atk_grade': atk_g,
        'hp_grade': hp_g,
        'spd_grade': spd_g,
        'hero_id': hero_id,
    }])

    delta_pred = model.predict(feature_row)[0]
    baseline_price = hero_tier_avg_price.get((hero_name, tier_score), df['resale_price'].median())
    predicted_price = baseline_price + delta_pred
    return predicted_price, baseline_price

# Buying recommendation
def should_buy(hero_name, tier, atk, hp, spd, auction_price):
    predicted, baseline = predict_resale_price(hero_name, tier, atk, hp, spd)
    margin = (predicted - auction_price) / auction_price
    print(f"\nPredicted resale price: {predicted:.2f} starry gems")
    print(f"Baseline price for {hero_name} ({tier}): {baseline:.2f} starry gems")
    print(f"Auction price: {auction_price:.2f} starry gems")
    print(f"Expected profit margin: {margin*100:.2f}%")
    if margin >= 0.05:
        print("Recommendation: BUY (≥5% profit expected)\n")
    else:
        print("Recommendation: SKIP (too risky)\n")

if __name__ == "__main__":
    hero_name = input("Enter hero name: ")
    tier = input("Enter tier (B+, A-, A, A+): ")
    atk = int(input("Enter attack stat: "))
    hp = int(input("Enter HP stat: "))
    spd = int(input("Enter speed stat (in 100s): "))
    auction_price = float(input("Enter current auction price: "))

    should_buy(hero_name, tier, atk, hp, spd, auction_price)
