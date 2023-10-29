import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data from Excel
df = pd.read_csv('sorted_forest_percentage.csv')



X = df[["year"]][1:10]  # Features
y = df["forest_area_percentage"][1:10]  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the linear regression model
model = LinearRegression()


# Train the model using the training data
model.fit(X_train, y_train)


# Predict forest area for a future year
future_year = 2023  # You can change this to your desired year
future_forest_area = model.predict([[future_year]])
print(f"Predicted forest area for {future_year}: {future_forest_area[0]}")

y_pred = model.predict(X_test)



# import matplotlib.pyplot as plt


# # Assuming you already have the model and X_encoded

# # Predict FTE_Staff
# y_pred = model.predict(X_top_10)

# # Create a scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X_top_10, y_top_10, alpha=0.5)
# # plt.plot(X_top_10, slope * X_top_10 + intercept, color='red', label='Regression Line')
# plt.title('Year vs. Forest Area')
# plt.xlabel('Year')
# plt.ylabel('Forest Area')
# plt.show()