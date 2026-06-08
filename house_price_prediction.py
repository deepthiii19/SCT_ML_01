# ============================================================
#  House Price Prediction — Linear Regression
#  SkillCraft Technology | Task 01
#  Features: GrLivArea (sq ft), BedroomAbvGr, FullBath, HalfBath
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. Load Data ─────────────────────────────────────────────
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print("=" * 55)
print("  HOUSE PRICE PREDICTION — LINEAR REGRESSION")
print("=" * 55)
print(f"\n📦 Training samples : {len(train_df)}")
print(f"📦 Test samples     : {len(test_df)}")

# ── 2. Feature Selection ─────────────────────────────────────
# Task requires: square footage, bedrooms, bathrooms
FEATURES = [
    "GrLivArea",       # Above-grade living area (sq ft)  → square footage
    "TotalBsmtSF",     # Total basement sq ft             → square footage
    "BedroomAbvGr",    # Bedrooms above grade
    "FullBath",        # Full bathrooms above grade
    "HalfBath",        # Half bathrooms above grade
]
TARGET = "SalePrice"

# ── 3. Preprocessing ─────────────────────────────────────────
df = train_df[FEATURES + [TARGET]].copy()

# Fill any missing values with column median
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"\n📊 Feature Summary:")
print(df[FEATURES].describe().round(2).to_string())

# ── 4. EDA Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("EDA — Feature vs Sale Price", fontsize=15, fontweight="bold")

feature_labels = {
    "GrLivArea"    : "Living Area (sq ft)",
    "TotalBsmtSF"  : "Basement Area (sq ft)",
    "BedroomAbvGr" : "Bedrooms",
    "FullBath"     : "Full Bathrooms",
    "HalfBath"     : "Half Bathrooms",
}

for ax, feat in zip(axes.flatten(), FEATURES):
    ax.scatter(df[feat], df[TARGET], alpha=0.4, color="#4C72B0", edgecolors="none", s=20)
    ax.set_xlabel(feature_labels[feat])
    ax.set_ylabel("Sale Price ($)")
    ax.set_title(f"{feature_labels[feat]} vs Price")
    # Trend line
    m, b = np.polyfit(df[feat], df[TARGET], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5)

# Correlation heatmap in last subplot
axes[1][2].set_visible(False)
fig.delaxes(axes[1][2])
ax_corr = fig.add_subplot(2, 3, 6)
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax_corr, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax_corr.set_title("Correlation Matrix")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA plot saved → eda_plots.png")

# ── 5. Train / Validation Split ──────────────────────────────
X = df[FEATURES].values
y = df[TARGET].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

# ── 6. Train Model ───────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("\n📐 Model Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"   {feat:<20} {coef:>12,.2f}")
print(f"   {'Intercept':<20} {model.intercept_:>12,.2f}")

# ── 7. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_val_sc)

mae  = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2   = r2_score(y_val, y_pred)

print("\n" + "─" * 40)
print("  VALIDATION METRICS")
print("─" * 40)
print(f"  MAE  (Mean Abs Error)  : ${mae:>10,.2f}")
print(f"  RMSE (Root MSE)        : ${rmse:>10,.2f}")
print(f"  R²   (R-squared)       :  {r2:>10.4f}")
print("─" * 40)

# ── 8. Residual & Prediction Plots ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_val, y_pred, alpha=0.5, color="#2ecc71", edgecolors="none", s=20)
lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title(f"Actual vs Predicted  (R² = {r2:.4f})")
axes[0].legend()

# Residuals
residuals = y_val - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color="#e74c3c", edgecolors="none", s=20)
axes[1].axhline(0, color="black", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Evaluation plot saved → model_evaluation.png")

# ── 9. Predict on Test Set & Save Submission ─────────────────
test_X = test_df[FEATURES].copy()
test_X.fillna(test_X.median(numeric_only=True), inplace=True)

test_X_sc    = scaler.transform(test_X.values)
test_preds   = model.predict(test_X_sc)

submission = pd.DataFrame({
    "Id"        : test_df["Id"],
    "SalePrice" : np.round(test_preds, 2)
})
submission.to_csv("submission.csv", index=False)

print(f"✅ Submission saved  → submission.csv  ({len(submission)} rows)")
print("\n🎉 All done! Check the 3 output files.")
print("=" * 55)