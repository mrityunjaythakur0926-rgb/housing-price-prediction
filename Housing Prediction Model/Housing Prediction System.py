import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. DATA GENERATION
# ==========================================
# Creating a mock dataset representing housing features
data = {
    'Area_sqft': [1500, 2000, 850, 1200, 2500, 3000, 1800, 900, 2200, 1100],
    'Bedrooms': [3, 4, 2, 2, 4, 5, 3, 2, 4, 2],
    'Age_years': [10, 5, 20, 15, 2, 1, 12, 25, 8, 18],
    'Location_Score': [8, 9, 5, 6, 9, 10, 7, 4, 8, 6], # 1-10 scale
    'Price_Lakhs': [75, 120, 40, 55, 150, 200, 85, 35, 130, 50]
}

df = pd.DataFrame(data)
print("Housing Data Preview:")
print(df.head())

# ==========================================
# 2. DATA ANALYSIS & VISUALIZATION
# ==========================================
# Correlation Matrix to see relationships
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# ==========================================
# 3. PREPROCESSING & SPLIT
# ==========================================
# Features (X) and Target (y)
X = df[['Area_sqft', 'Bedrooms', 'Age_years', 'Location_Score']]
y = df['Price_Lakhs']

# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL TRAINING (Linear Regression)
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
print(f"R2 Score (Accuracy): {r2:.4f}")

# ==========================================
# 6. INTERACTIVE PREDICTION INTERFACE
# ==========================================
def predict_price(area, bedrooms, age, loc_score):
    features = np.array([[area, bedrooms, age, loc_score]])
    price = model.predict(features)[0]
    return price

print("\n--- Live Prediction Example ---")
# Example: 1600 sqft, 3 Bedrooms, 5 years old, Location Score 8
est_area = 1600
est_bed = 3
est_age = 5
est_loc = 8

predicted_value = predict_price(est_area, est_bed, est_age, est_loc)
print(f"House Details: {est_area} sqft, {est_bed} BHK, {est_age} years old")
print(f"Predicted Price: ₹{predicted_value:.2f} Lakhs")

# Plotting Actual vs Predicted (for visual confirmation)
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.show()