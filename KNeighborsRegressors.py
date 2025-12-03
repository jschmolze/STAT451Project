import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

URL = "C:/VISP_LEC/STAT 451/project/Merged_WB_PWT_DEMO.csv"
df = pd.read_csv(URL)

# Select numeric feature columns excluding identifiers and target
features = [c for c in df.columns 
            if c not in ['country', 'year', 'gdp_growth'] 
            and pd.api.types.is_numeric_dtype(df[c])]

# print("Numeric feature columns:", features)

# Apply mean imputation to numeric features
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Compute GDP growth if rgdpe column exists
if 'rgdpe' in df.columns:
    df['gdp_growth'] = df.groupby('country')['rgdpe'].pct_change(fill_method=None)
else:
    raise KeyError("The dataset does not contain 'rgdpe' column, cannot compute gdp_growth")

df = df.dropna(subset=['gdp_growth']).copy()

X = df[features]
y = df['gdp_growth']

tscv = TimeSeriesSplit(n_splits=5)
# print("Time split -> Train size:", X_train.shape, "Test size:", X_test.shape, "Test year:", last_year)

# ---------- Train Model ----------
knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='mean')),  # Mean imputation
    ("scaler", StandardScaler()),                 # Standardization
    ("knn", KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

scores = cross_val_score(
    knn_pipeline, X, y,
    cv=tscv,
    scoring='r2'
)

fold = 1
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    knn_pipeline.fit(X_train, y_train)
    y_pred = knn_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"Fold {fold}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    print(f"  MSE={mse:.4f}, R²={r2:.4f}")
    fold += 1
# Fit model
knn_pipeline.fit(X_train, y_train)

# Predict
y_pred = knn_pipeline.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
