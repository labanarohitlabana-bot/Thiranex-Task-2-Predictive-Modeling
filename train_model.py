"""
Thiranex Task 2: Predictive Modeling Using Machine Learning
Demo project: predict student final scores from study-related features.

IMPORTANT:
- The included dataset is synthetic/simulated for practice, not real student records.
- Replace the CSV with an approved real dataset only if your internship mentor permits it.
- This is a regression task, so MAE/RMSE/R² and regression plots are appropriate;
  confusion matrices and ROC curves are for classification tasks.
"""
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
target = "final_score"
X, y = df.drop(columns=[target]), df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Random Forest Regressor": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2
        )),
    ]),
}

results, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(OUT / "model_metrics.csv", index=False)
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
best_pred = predictions[best_name]

pd.DataFrame({
    "actual_final_score": y_test.to_numpy(),
    "predicted_final_score": best_pred,
}).to_csv(OUT / "test_predictions.csv", index=False)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_pred, alpha=0.7)
low = min(y_test.min(), best_pred.min())
high = max(y_test.max(), best_pred.max())
plt.plot([low, high], [low, high], "--")
plt.xlabel("Actual final score")
plt.ylabel("Predicted final score")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

residuals = y_test.to_numpy() - best_pred
plt.figure(figsize=(7, 5))
plt.scatter(best_pred, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted final score")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "residual_plot.png", dpi=160)
plt.close()

if best_name == "Random Forest Regressor":
    values = best_model.named_steps["model"].feature_importances_
else:
    values = abs(best_model.named_steps["model"].coef_)
importance = pd.DataFrame({"Feature": X.columns, "Importance": values})
importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(OUT / "feature_importance.csv", index=False)

plt.figure(figsize=(7, 5))
plt.barh(importance["Feature"][::-1], importance["Importance"][::-1])
plt.xlabel("Importance (relative)")
plt.title(f"Feature Importance — {best_name}")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=160)
plt.close()

print("Dataset rows:", len(df))
print("Missing values before imputation:")
print(df.isna().sum())
print("\nModel comparison:")
print(results_df.to_string(index=False))
print("\nBest model by RMSE:", best_name)
print("Saved metrics, predictions, and plots to:", OUT)
