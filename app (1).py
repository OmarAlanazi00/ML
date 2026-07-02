import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide",
)

sns.set_theme(style="whitegrid")
DEPARTMENTS = ["Arts", "Business", "Engineering", "Medical", "Science"]

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
DEFAULT_PATHS = [
    "student_lifestyle_100k.csv",
    "data/student_lifestyle_100k.csv",
]


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def get_dataset():
    for p in DEFAULT_PATHS:
        try:
            return load_data(p), p
        except FileNotFoundError:
            continue
    return None, None


# ------------------------------------------------------------------
# Full preprocessing + training pipeline (mirrors the notebook)
# ------------------------------------------------------------------
@st.cache_resource
def build_pipeline(df: pd.DataFrame):
    df = df.copy()

    # ---- Feature engineering (same as notebook) ----
    def sleep_category(hours):
        if hours < 6:
            return "Short"
        elif hours <= 8:
            return "Normal"
        else:
            return "Long"

    df["sleep_category"] = df["Sleep_Duration"].apply(sleep_category)
    df["study_balance"] = df["Study_Hours"] - df["Social_Media_Hours"]

    # ---- Encoding ----
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Depression"] = df["Depression"].astype(int)
    df = pd.get_dummies(df, columns=["Department"], dtype=int)
    df = pd.get_dummies(df, columns=["sleep_category"], dtype=int)

    # ---- Features / target ----
    X = df.drop(columns=["Depression", "Student_ID"])
    y = df["Depression"]
    feature_columns = X.columns.tolist()

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Scaling ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- PCA ----
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # ---- Models ----
    lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    lr_model.fit(X_train_pca, y_train)
    lr_pred = lr_model.predict(X_test_pca)

    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_model.fit(X_train_pca, y_train)
    rf_pred = rf_model.predict(X_test_pca)

    metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, lr_pred),
            "Precision": precision_score(y_test, lr_pred),
            "Recall": recall_score(y_test, lr_pred),
            "F1-score": f1_score(y_test, lr_pred),
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred),
            "Recall": recall_score(y_test, rf_pred),
            "F1-score": f1_score(y_test, rf_pred),
        },
    }

    return {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "pca": pca,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "y_test": y_test,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred,
        "metrics": metrics,
        "X_train": X_train,
        "engineered_df": df,
    }


def prepare_single_input(raw: dict, feature_columns: list):
    """Turn a dict of raw user inputs into a model-ready row (same encoding as training)."""
    sleep = raw["Sleep_Duration"]
    if sleep < 6:
        sleep_cat = "Short"
    elif sleep <= 8:
        sleep_cat = "Normal"
    else:
        sleep_cat = "Long"

    row = {
        "Age": raw["Age"],
        "Gender": 1 if raw["Gender"] == "Male" else 0,
        "CGPA": raw["CGPA"],
        "Sleep_Duration": raw["Sleep_Duration"],
        "Study_Hours": raw["Study_Hours"],
        "Social_Media_Hours": raw["Social_Media_Hours"],
        "Physical_Activity": raw["Physical_Activity"],
        "Stress_Level": raw["Stress_Level"],
        "study_balance": raw["Study_Hours"] - raw["Social_Media_Hours"],
    }
    for d in DEPARTMENTS:
        row[f"Department_{d}"] = 1 if raw["Department"] == d else 0
    for c in ["Long", "Normal", "Short"]:
        row[f"sleep_category_{c}"] = 1 if sleep_cat == c else 0

    row_df = pd.DataFrame([row])
    row_df = row_df.reindex(columns=feature_columns, fill_value=0)
    return row_df, sleep_cat


# ==================================================================
# APP LAYOUT
# ==================================================================
st.title("🧠 Student Depression Prediction App")
st.caption(
    "An interactive walkthrough of a full ML workflow: EDA, cleaning, "
    "preprocessing, PCA, modeling, evaluation, and live prediction."
)

df, source_path = get_dataset()

if df is None:
    st.warning(
        "Dataset not found automatically. Please upload the "
        "`student_lifestyle_100k.csv` file to run the app."
    )
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        st.stop()

tab_overview, tab_eda, tab_clean, tab_model, tab_predict = st.tabs(
    [
        "📋 Data Overview",
        "📊 EDA",
        "🧹 Cleaning & Preprocessing",
        "🤖 Model Results",
        "🔮 Interactive Prediction",
    ]
)

# ------------------------------------------------------------------
# TAB 1: Data Overview
# ------------------------------------------------------------------
with tab_overview:
    st.header("Data Overview")
    st.markdown(
        """
This dataset contains lifestyle and academic information for **100,000 students**,
used to predict whether a student is at risk of **Depression** (binary classification).

**Columns:**

| Column | Description |
|---|---|
| `Student_ID` | Unique identifier for each student |
| `Age` | Student age (18–24) |
| `Gender` | Male / Female |
| `Department` | Academic department (Science, Engineering, Medical, Arts, Business) |
| `CGPA` | Cumulative grade point average |
| `Sleep_Duration` | Average hours of sleep per day |
| `Study_Hours` | Average hours spent studying per day |
| `Social_Media_Hours` | Average hours spent on social media per day |
| `Physical_Activity` | Minutes of physical activity |
| `Stress_Level` | Self-reported stress level |
| `Depression` | Target variable — whether the student is at risk of depression |
        """
    )
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    st.subheader("Sample rows")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# ------------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Depression distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Depression", hue="Depression", legend=False, ax=ax)
    ax.set_title("Depression Distribution")
    st.pyplot(fig)
    st.caption(
        "Shows how many students are classified as depressed vs. not — "
        "useful to spot class imbalance."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Age"], bins=15, kde=True, color="skyblue", ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Sleep duration by Depression")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df, x="Depression", y="Sleep_Duration",
            hue="Depression", legend=False, palette="coolwarm", ax=ax
        )
        ax.set_title("Sleep Duration by Depression")
        st.pyplot(fig)
        st.caption("Students marked as depressed tend to sleep fewer hours.")

    st.subheader("Study hours by Depression")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df, x="Depression", y="Study_Hours",
        hue="Depression", legend=False, palette="Set3", ax=ax
    )
    ax.set_title("Study Hours by Depression")
    st.pyplot(fig)
    st.caption("Study hours show little separation between the two groups.")

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True, cmap="YlGnBu", ax=ax
    )
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 3: Cleaning & Preprocessing
# ------------------------------------------------------------------
with tab_clean:
    st.header("Data Cleaning")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Missing values per column**")
        st.dataframe(df.isnull().sum().rename("missing_count"))
    with c2:
        st.write("**Duplicate rows**")
        st.metric("Duplicated rows", int(df.duplicated().sum()))
    st.info(
        "No missing values and no duplicate rows were found, so no imputation "
        "or row-dropping was necessary. Outliers (e.g. in Age, CGPA) were "
        "inspected with boxplots and kept, since they reflect plausible "
        "real-world values rather than data errors."
    )

    st.header("Feature Engineering")
    st.markdown(
        """
- **`sleep_category`** — bucketed from `Sleep_Duration`: `< 6h → Short`,
  `6–8h → Normal`, `> 8h → Long`.
- **`study_balance`** — `Study_Hours - Social_Media_Hours`, a simple proxy
  for whether a student studies more than they use social media.
        """
    )

    st.header("Encoding & Scaling")
    st.markdown(
        """
- **`Gender`** → binary map (`Male=1`, `Female=0`).
- **`Department`** and **`sleep_category`** → one-hot encoded (no ordinal
  relationship between categories).
- **Feature scaling (`StandardScaler`)** is applied because Logistic
  Regression is sensitive to feature magnitude (its coefficients and
  regularization penalize larger-scale features more), and distance/variance
  based steps like PCA also require standardized inputs to work correctly.
        """
    )

    pipe = build_pipeline(df)
    st.subheader("Before / after encoding (sample)")
    b1, b2 = st.columns(2)
    with b1:
        st.caption("Before")
        st.dataframe(df[["Gender", "Department", "Sleep_Duration"]].head())
    with b2:
        st.caption("After")
        preview_cols = [c for c in pipe["feature_columns"] if c in pipe["engineered_df"].columns]
        st.dataframe(pipe["engineered_df"][preview_cols].head())

    st.header("Dimensionality Reduction (PCA)")
    n_orig = pipe["X_train"].shape[1]
    n_pca = pipe["pca"].n_components_
    st.write(f"Original features: **{n_orig}** → PCA components (95% variance): **{n_pca}**")
    fig, ax = plt.subplots(figsize=(7, 4))
    cum = pipe["pca"].explained_variance_ratio_.cumsum()
    ax.plot(range(1, len(cum) + 1), cum, marker="o")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.grid(True)
    st.pyplot(fig)
    st.caption(
        f"PCA reduced the feature space from {n_orig} to {n_pca} components while "
        "retaining 95% of the variance, simplifying the model input with minimal "
        "information loss."
    )

# ------------------------------------------------------------------
# TAB 4: Model Results
# ------------------------------------------------------------------
with tab_model:
    st.header("Model Results")
    pipe = build_pipeline(df)
    metrics_df = pd.DataFrame(pipe["metrics"]).T
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    best_model = metrics_df["F1-score"].idxmax()
    st.success(f"Best model by F1-score: **{best_model}**")

    model_choice = st.radio(
        "Show confusion matrix for:", ["Logistic Regression", "Random Forest"], horizontal=True
    )
    pred = pipe["lr_pred"] if model_choice == "Logistic Regression" else pipe["rf_pred"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(pipe["y_test"], pred, ax=ax)
    ax.set_title(f"{model_choice} Confusion Matrix")
    st.pyplot(fig)

    st.markdown(
        """
**Conclusion:** Both Logistic Regression and Random Forest were trained on the
PCA-reduced, scaled feature set with `class_weight="balanced"` to account for
class imbalance in the Depression target. Metrics above (accuracy, precision,
recall, F1) and the confusion matrix show how well each model separates the
two classes. With more time, useful next steps would include hyperparameter
tuning (e.g. `GridSearchCV`), trying additional models (e.g. gradient boosting),
and testing alternative resampling strategies (SMOTE) for the imbalance.
        """
    )

# ------------------------------------------------------------------
# TAB 5: Interactive Prediction
# ------------------------------------------------------------------
with tab_predict:
    st.header("Interactive Prediction")
    st.write("Enter student details to get a live depression-risk prediction.")

    pipe = build_pipeline(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 24, 21)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", DEPARTMENTS)
    with c2:
        cgpa = st.slider("CGPA", 1.0, 4.0, 2.9, 0.01)
        sleep_hours = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
        study_hours = st.slider("Study Hours", 0.0, 13.0, 4.5, 0.1)
    with c3:
        social_hours = st.slider("Social Media Hours", 0.0, 10.0, 3.5, 0.1)
        physical_activity = st.slider("Physical Activity (min)", 0, 150, 74)
        stress_level = st.slider("Stress Level", 2, 10, 4)

    model_pick = st.radio(
        "Model to use for prediction:",
        ["Random Forest", "Logistic Regression"],
        horizontal=True,
    )

    if st.button("Predict", type="primary"):
        raw = {
            "Age": age, "Gender": gender, "Department": department,
            "CGPA": cgpa, "Sleep_Duration": sleep_hours, "Study_Hours": study_hours,
            "Social_Media_Hours": social_hours, "Physical_Activity": physical_activity,
            "Stress_Level": stress_level,
        }
        row_df, sleep_cat = prepare_single_input(raw, pipe["feature_columns"])
        row_scaled = pipe["scaler"].transform(row_df)
        row_pca = pipe["pca"].transform(row_scaled)

        model = pipe["rf_model"] if model_pick == "Random Forest" else pipe["lr_model"]
        pred = model.predict(row_pca)[0]
        proba = model.predict_proba(row_pca)[0][1]

        st.caption(f"Derived features → sleep_category: **{sleep_cat}**, "
                   f"study_balance: **{study_hours - social_hours:.1f}**")

        if pred == 1:
            st.error(f"⚠️ Predicted: At risk of Depression (probability: {proba:.1%})")
        else:
            st.success(f"✅ Predicted: Not at risk of Depression (probability of risk: {proba:.1%})")
