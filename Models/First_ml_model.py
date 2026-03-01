# Step 1: Import library
from statistics import LinearRegression

import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 2: Create sample dataset
data = {
    "area": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 1200, 1800, 2200, 2800, 3200],
    "price": [300000, 400000, 500000, 600000, 700000, 810000, 895000, 1020000, 1150000, 345000, 480000, 560000, 675000, 740000]
}

# Step 3: Using pandas → We created table of the Dataset
df = pd.DataFrame(data)

# Step 4: Separate input (X) and output (Y)
X = df[["area"]]   # Input
y = df["price"]    # Output

# Step 5: Create MODEL
model = LinearRegression()

# Step 6: Train model
model.fit(X, y)

# Step 7: Predict now house price
new_area = [[int(input("Enter the new area: "))]]
predicted_price = model.predict(new_area)
print("Predicted price: ", predicted_price[0])
