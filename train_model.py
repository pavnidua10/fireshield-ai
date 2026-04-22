import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import xgboost as xgb
import pickle

# ─────────────────────────────────────────────
# STEP 1: Load Real UCI Forest Fires Dataset
# ─────────────────────────────────────────────
def load_real_data(csv_path='forestfires.csv'):
    df = pd.read_csv(csv_path)
    print(f"Loaded real dataset: {df.shape[0]} records, {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────────
# STEP 2: Convert area burned → risk class
#   0 ha burned       → 0 (Low Risk)
#   0 < area ≤ 25 ha  → 1 (Medium Risk)
#   area > 25 ha      → 2 (High Risk)
# ─────────────────────────────────────────────
def assign_risk_label(area):
    if area == 0:
        return 0
    elif area <= 25:
        return 1
    else:
        return 2

# ─────────────────────────────────────────────
# STEP 3: Engineer the 11 features your app uses
#
# Real features from UCI:
#   temp  → Temperature
#   RH    → Humidity
#   wind  → Wind_Speed
#   rain  → Rainfall_7d (approximated)
#   FFMC  → Fine Fuel Moisture (used to derive soil moisture proxy)
#   DC    → Drought Code (used to derive fire history proxy)
#   ISI   → Initial Spread Index (used to derive leaf litter proxy)
#   DMC   → Duff Moisture Code (used to derive forest cover proxy)
#
# Estimated features (no real data available, smartly derived):
#   Forest_Cover   → derived from DMC (deeper organic layer = more forest)
#   Soil_Moisture  → derived from FFMC inverted (low FFMC = dry = low moisture)
#   NDVI           → derived from forest cover + soil moisture (same formula as app)
#   Fire_History   → derived from DC (long-term drought index = historical fire risk)
#   Leaf_Litter    → derived from ISI + dryness (same formula as app)
#   Elevation      → randomly sampled realistic range (not in UCI dataset)
#   Road_Distance  → randomly sampled realistic range (not in UCI dataset)
# ─────────────────────────────────────────────
def engineer_features(df):
    n = len(df)
    np.random.seed(42)

    # ── Direct mappings ──
    temperature   = df['temp'].values
    humidity      = df['RH'].values
    wind_speed    = df['wind'].values
    # UCI rain is daily, multiply by 7 to approximate 7-day sum
    rainfall_7d   = (df['rain'] * 7).values

    # ── Soil Moisture from FFMC ──
    # FFMC range: 0–96. Higher FFMC = drier fine fuels = lower soil moisture
    ffmc_norm     = df['FFMC'].values / 96.0           # 0 → 1
    soil_moisture = np.clip((1 - ffmc_norm) * 100, 0, 100)  # invert → moisture %

    # ── Forest Cover from DMC ──
    # DMC range: 0–291 in this dataset. Higher DMC = more organic matter = denser forest
    dmc_norm      = np.clip(df['DMC'].values / 200.0, 0, 1)
    forest_cover  = dmc_norm * 100   # scale to 0–100%

    # ── NDVI (same formula as app.py) ──
    ndvi = np.clip(
        (forest_cover / 100.0) * 0.7 + (soil_moisture / 100.0) * 0.3,
        0.0, 1.0
    )

    # ── Leaf Litter Index (same formula as app.py) ──
    leaf_litter = np.clip(
        (forest_cover / 100.0) * 0.6 + ((100 - soil_moisture) / 100.0) * 0.4,
        0.0, 1.0
    )

    # ── Fire History from DC ──
    # DC range: 0–860. Higher DC = longer drought = more historical fire risk
    dc_norm       = np.clip(df['DC'].values / 800.0, 0, 1)
    fire_history  = dc_norm * 10   # scale to 0–10

    # ── Elevation: realistic random distribution ──
    # Montesinho park is ~700–1400m elevation
    elevation = np.random.normal(900, 200, n)
    elevation = np.clip(elevation, 500, 1500)

    # ── Road Distance: realistic random distribution ──
    road_distance = np.random.exponential(3000, n)
    road_distance = np.clip(road_distance, 100, 15000)

    X = pd.DataFrame({
        'Temperature':      temperature,
        'Humidity':         humidity,
        'Wind_Speed':       wind_speed,
        'Rainfall_7d':      rainfall_7d,
        'Soil_Moisture':    soil_moisture,
        'Forest_Cover':     forest_cover,
        'NDVI':             ndvi,
        'Fire_History':     fire_history,
        'Elevation':        elevation,
        'Road_Distance':    road_distance,
        'Leaf_Litter_Index': leaf_litter
    })

    return X

# ─────────────────────────────────────────────
# STEP 4: Balance classes using oversampling
# Problem: 247 Low, 218 Medium, only 52 High
# Fix: Upsample Medium and High to match Low
# ─────────────────────────────────────────────
def balance_classes(X, y):
    df_combined = X.copy()
    df_combined['label'] = y

    df_low    = df_combined[df_combined['label'] == 0]
    df_med    = df_combined[df_combined['label'] == 1]
    df_high   = df_combined[df_combined['label'] == 2]

    target = max(len(df_low), len(df_med), len(df_high))

    df_med_up  = resample(df_med,  replace=True, n_samples=target, random_state=42)
    df_high_up = resample(df_high, replace=True, n_samples=target, random_state=42)

    df_balanced = pd.concat([df_low, df_med_up, df_high_up])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    y_bal = df_balanced['label'].values
    X_bal = df_balanced.drop('label', axis=1)

    print(f"\nAfter balancing: {len(df_balanced)} total samples")
    print(f"  Low:    {(y_bal == 0).sum()}")
    print(f"  Medium: {(y_bal == 1).sum()}")
    print(f"  High:   {(y_bal == 2).sum()}")

    return X_bal, y_bal

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  FireShield AI — Real Data Model Trainer")
    print("=" * 50)

    # 1. Load
    df = load_real_data('forestfires.csv')

    # 2. Labels
    y = np.array([assign_risk_label(a) for a in df['area']])
    print(f"\nRisk class distribution (raw):")
    print(f"  Low  (0): {(y == 0).sum()}")
    print(f"  Med  (1): {(y == 1).sum()}")
    print(f"  High (2): {(y == 2).sum()}")

    # 3. Features
    X = engineer_features(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")

    # 4. Balance
    X, y = balance_classes(X, y)

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 6. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 7. Train XGBoost
    print("\nTraining XGBoost Classifier on real data...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train_scaled, y_train)

    # 8. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # 9. Feature importance
    importances = model.feature_importances_
    feat_names  = X.columns.tolist()
    print("\nFeature Importances:")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"  {name:<22} {bar} {imp:.4f}")

    # 10. Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✅ Saved new model.pkl and scaler.pkl successfully!")
    print("   Your app.py will automatically use these on next run.")