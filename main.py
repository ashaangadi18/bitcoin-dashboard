import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1️⃣ Load your dataset
df = pd.read_csv("Bitcoin.csv")  # replace with your dataset name if different

# 2️⃣ Select features (X) and target (y)
# Adjust feature columns if needed
feature_cols = ['Open', 'High', 'Low', 'Volume']  # input features
X = df[feature_cols]
y = df['Close']  # target: closing price

# 3️⃣ Split into train and test (optional, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Prepare input for prediction (example: latest values)
# Replace with real latest Bitcoin values
latest_values = [[39950, 40500, 39800, 36000]]  # [Open, High, Low, Volume]

# Convert to DataFrame with the same column names to avoid warnings
input_df = pd.DataFrame(latest_values, columns=feature_cols)

# 6️⃣ Predict
prediction = model.predict(input_df)
print("Tomorrow's Bitcoin Price Prediction:", prediction[0])
