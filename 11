import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("simple_house.csv")

print(df)

# Features and Target
X = df[['area']]
y = df['price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Train model on full dataset for visualization
model.fit(X, y)
y_pred_full = model.predict(X)

# Plot graph
plt.scatter(X, y)
plt.plot(X, y_pred_full)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("y = mx + c (Linear Regression)")

plt.show()