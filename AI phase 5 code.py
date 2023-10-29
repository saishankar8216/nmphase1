import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate a sample dataset (replace this with your actual data)
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100) * 2  # Simulated energy consumption

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict energy consumption for test set
y_pred = rf_regressor.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Visualize the results
plt.figure(figsize=(12, 4))

# Scatter Plot
plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Energy Consumption')
plt.title('Actual vs. Predicted')
plt.legend()

# Bar Chart
plt.subplot(1, 3, 2)
labels = ['Actual', 'Predicted']
values = [np.sum(y_test), np.sum(y_pred)]
plt.bar(labels, values, color=['blue', 'red'])
plt.xlabel('Category')
plt.ylabel('Total Energy Consumption')
plt.title('Total Energy Consumption')

# Pie Chart
plt.subplot(1, 3, 3)
sizes = [np.sum(y_test), np.sum(y_pred)]
colors = ['blue', 'red']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Energy Consumption Distribution')

plt.tight_layout()
plt.show()
