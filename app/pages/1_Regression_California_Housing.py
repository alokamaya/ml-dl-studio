import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="California Housing", layout="wide")

st.title("🏠 California Housing Price Prediction")

st.markdown("""
### 🎯 Objective
Build a regression model to predict house prices using socio-economic and location-based features.

### 🧠 What I Learned
- Feature vs Target separation
- Train-test split
- Linear Regression fundamentals
- Model evaluation (MAE, RMSE, R²)
- Feature scaling using StandardScaler
- Building ML Pipelines
""")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.rename(columns={"MedHouseVal": "PRICE"}, inplace=True)
    return df

df = load_data()

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📂 Dataset Overview")
col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)

with col2:
    st.write("Columns:", list(df.columns))

st.dataframe(df.head())

# -------------------------------
# EDA Insights
# -------------------------------
st.subheader("🔍 Key Insights")

st.markdown("""
- **MedInc (Median Income)** shows strong positive correlation with house prices  
- **Population & AveOccup** show weaker relationships  
- Dataset is clean (no missing values)  
""")

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("📊 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -------------------------------
# Train Model
# -------------------------------
st.subheader("⚙️ Model Training")

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# -------------------------------
# Metrics
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("R² Score", f"{r2:.3f}")

# -------------------------------
# Actual vs Predicted Plot
# -------------------------------
st.subheader("📈 Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)

# -------------------------------
# Interpretation
# -------------------------------
st.subheader("🧠 Interpretation")

st.markdown(f"""
- Average prediction error ≈ **{mae:.2f} (~$50k)**  
- Model explains **~{r2*100:.1f}% of variance**  
- Linear Regression gives a **moderate baseline performance**  

👉 Improvement needed using advanced models.
""")

# -------------------------------
# Try Prediction
# -------------------------------
st.subheader("🔮 Try Prediction")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction * 100000:.2f}")



st.subheader("📓 Download Notebook")

with open("notebooks/regression/regression.ipynb", "rb") as f:
    st.download_button(
        label="⬇️ Download Regression Notebook",
        data=f,
        file_name="regression.ipynb",
        mime="application/octet-stream"
    )

# -------------------------------
# Future Work
# -------------------------------
st.subheader("🚀 Future Improvements")

st.markdown("""
- Add Ridge & Lasso Regression  
- Use Random Forest / Gradient Boosting  
- Feature engineering  
- Model comparison dashboard  
""")