# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.title(" California Housing Dashboard")
st.sidebar.header("Configuration")

# Load data
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame

df = load_data()

# Display data
if st.checkbox("Show raw data"):
    st.write(df.head(100))

# Select model
model_type = st.sidebar.selectbox(
    "Choose Regression Model",
    ["Linear", "Ridge", "Lasso"]
)

alpha = st.sidebar.slider("Alpha (for Ridge/Lasso)", 0.01, 10.0, 1.0)

# Train model
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == "Linear":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.subheader(f" {model_type} Regression Results")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
col2.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
col3.metric("Training Samples", len(X_train))

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"{model_type} Regression: Actual vs Predicted")
st.pyplot(fig)

# Display feature importance
if model_type == "Linear":
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    st.bar_chart(importance.set_index('Feature')['Coefficient'])