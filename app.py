import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model

# Load model and data
model, data = train_model()

st.title("🌱 Crop Yield Prediction System")

st.sidebar.header("Enter Input Values")

# Inputs
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 150)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 28)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 70)
nitrogen = st.sidebar.slider("Nitrogen", 0, 100, 40)
phosphorus = st.sidebar.slider("Phosphorus", 0, 100, 50)
potassium = st.sidebar.slider("Potassium", 0, 100, 60)

# Prediction
if st.sidebar.button("Predict Yield"):
    input_data = [[rainfall, temperature, humidity, nitrogen, phosphorus, potassium]]
    prediction = model.predict(input_data)
    st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")

# ------------------ GRAPHS ------------------

st.subheader("📊 Data Visualization")

# Rainfall vs Yield
st.write("Rainfall vs Yield")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=data["rainfall"], y=data["yield"], ax=ax1)
st.pyplot(fig1)

# Temperature vs Yield
st.write("Temperature vs Yield")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=data["temperature"], y=data["yield"], ax=ax2)
st.pyplot(fig2)

# Heatmap
st.write("Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Distribution Plot
st.write("Yield Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(data["yield"], kde=True, ax=ax4)
st.pyplot(fig4)
