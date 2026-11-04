"""
AirSense AI - Indian Air Quality Prediction System
Models:
1. Linear Regression: predicts numerical AQI.
2. Decision Tree Classifier: predicts AQI_Bucket.

Place city_day.csv in the same folder as this file, then run:
    python airsense_ai.py
"""

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "city_day.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

data = pd.read_csv('/Users/jaswanthboyana/Desktop/archive (1)/city_day.csv')

print("\n========== AIRSENSE AI ==========")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nMissing values before preprocessing:")
print(data.isnull().sum())

# Keep only useful columns. Date is not needed for these two models.
pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]
numeric_features = [col for col in pollutants if col in data.columns]
categorical_features = ["City"] if "City" in data.columns else []
features = numeric_features + categorical_features

# ---------------------------------------------------------
# 2. DATA VISUALIZATION - THREE VISUALIZATIONS
# ---------------------------------------------------------

# Visualization 1: Average AQI of the 10 most polluted cities.
# Dictionary = hash map. It stores each city's AQI total and count.
city_totals = {}
city_counts = {}

for city, aqi in zip(data["City"], data["AQI"]):
    if pd.notna(city) and pd.notna(aqi):
        city_totals[city] = city_totals.get(city, 0) + aqi
        city_counts[city] = city_counts.get(city, 0) + 1

city_averages = {
    city: city_totals[city] / city_counts[city]
    for city in city_totals
}

# Sorting finds the 10 cities with the highest average AQI.
top_cities = sorted(
    city_averages.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]

city_names = [item[0] for item in top_cities]
city_aqi = [item[1] for item in top_cities]

plt.figure(figsize=(11, 6))
plt.barh(city_names, city_aqi)
plt.title("Top 10 Cities by Average AQI")
plt.xlabel("Average AQI (higher means more polluted)")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_1_city_aqi.png", dpi=300)
plt.show()

# Visualization 2: Correlation between numerical pollutants and AQI.
correlation_columns = numeric_features + (["AQI"] if "AQI" in data.columns else [])
correlation_data = data[correlation_columns].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(correlation_data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Pollutants and AQI")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_2_correlation.png", dpi=300)
plt.show()

# Visualization 4: Distribution of AQI values.
# A histogram shows how frequently AQI values occur in different ranges.
aqi_values = data["AQI"].dropna()

plt.figure(figsize=(9, 6))
sns.histplot(aqi_values, bins=30, kde=True)
plt.title("Distribution of AQI Values")
plt.xlabel("AQI")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_4_aqi_distribution.png", dpi=300)
plt.show()

# Visualization 5: Monthly average AQI trend.
# Convert Date to datetime, then group records by month.
date_data = data.dropna(subset=["Date", "AQI"]).copy()
date_data["Date"] = pd.to_datetime(date_data["Date"], errors="coerce")
date_data = date_data.dropna(subset=["Date"])
date_data["Month"] = date_data["Date"].dt.to_period("M").dt.to_timestamp()
monthly_aqi = date_data.groupby("Month")["AQI"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_aqi, x="Month", y="AQI")
plt.title("Monthly Average AQI Trend")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_5_monthly_aqi_trend.png", dpi=300)
plt.show()

# Visualization 6: Compare pollutant distributions.
# Box plots show the median, spread and possible outliers for each pollutant.
boxplot_columns = [
    col for col in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    if col in data.columns
]
boxplot_data = data[boxplot_columns].melt(
    var_name="Pollutant", value_name="Measured Value"
).dropna()

plt.figure(figsize=(11, 6))
sns.boxplot(data=boxplot_data, x="Pollutant", y="Measured Value")
plt.title("Distribution of Major Air Pollutants")
plt.xlabel("Pollutant")
plt.ylabel("Measured Value")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_6_pollutant_boxplot.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 3. PREPROCESSING FUNCTION
# ---------------------------------------------------------

def create_preprocessor():
    """Fill missing values and convert City names into numeric features."""

    numeric_pipeline = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode_city", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numbers", numeric_pipeline, numeric_features),
        ("city", categorical_pipeline, categorical_features)
    ])


# ---------------------------------------------------------
# 4. MODEL 1 - LINEAR REGRESSION (AQI PREDICTION)
# ---------------------------------------------------------

print("\n========== MODEL 1: LINEAR REGRESSION ==========")

regression_data = data.dropna(subset=["AQI"]).copy()
X_reg = regression_data[features]
y_reg = regression_data["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

regression_model = Pipeline(steps=[
    ("preprocessing", create_preprocessor()),
    ("model", LinearRegression())
])

regression_model.fit(X_train, y_train)
regression_predictions = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, regression_predictions)
mse = mean_squared_error(y_test, regression_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, regression_predictions)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared (R2):", round(r2, 3))

# Visualization 3: Actual AQI compared with predicted AQI.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, regression_predictions, alpha=0.35)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual vs Predicted AQI")

# A perfect prediction would lie on this diagonal line.
minimum = min(y_test.min(), regression_predictions.min())
maximum = max(y_test.max(), regression_predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualization_3_actual_vs_predicted.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. MODEL 2 - DECISION TREE CLASSIFICATION
# ---------------------------------------------------------

print("\n========== MODEL 2: DECISION TREE CLASSIFIER ==========")

classification_data = data.dropna(subset=["AQI_Bucket"]).copy()
X_class = classification_data[features]
y_class = classification_data["AQI_Bucket"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

classification_model = Pipeline(steps=[
    ("preprocessing", create_preprocessor()),
    ("model", DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ))
])

classification_model.fit(X_train_c, y_train_c)
class_predictions = classification_model.predict(X_test_c)
class_probabilities = classification_model.predict_proba(X_test_c)

accuracy = accuracy_score(y_test_c, class_predictions)
precision = precision_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)
recall = recall_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test_c, class_predictions, average="weighted", zero_division=0
)
loss = log_loss(
    y_test_c, class_probabilities,
    labels=classification_model.named_steps["model"].classes_
)

print("Accuracy:", round(accuracy, 3))
print("Precision (weighted):", round(precision, 3))
print("Recall (weighted):", round(recall, 3))
print("F1 Score (weighted):", round(f1, 3))
print("Logarithmic Loss:", round(loss, 3))

# AUC is calculated using a one-vs-rest approach for multiple classes.
try:
    auc = roc_auc_score(
        y_test_c,
        class_probabilities,
        multi_class="ovr",
        average="weighted",
        labels=classification_model.named_steps["model"].classes_
    )
    print("ROC AUC (weighted OVR):", round(auc, 3))
except ValueError:
    print("ROC AUC could not be calculated for this test split.")

# Confusion matrix is printed as a table (not an additional visualization).
class_names = classification_model.named_steps["model"].classes_
matrix = confusion_matrix(y_test_c, class_predictions, labels=class_names)
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(matrix, index=class_names, columns=class_names))

print("\nAll six visualizations have been saved in the 'outputs' folder.")
print("Project completed successfully.")
