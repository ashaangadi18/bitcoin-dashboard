from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset and train model once
df = pd.read_csv("Bitcoin.csv")  # your CSV
feature_cols = ['Open', 'High', 'Low', 'Volume']  # adjust to your CSV
X = df[feature_cols]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume = float(request.form['volume'])
    
    input_df = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=feature_cols)
    prediction = model.predict(input_df)
    
    return render_template('index.html', prediction_text=f"Tomorrow's Bitcoin Price: {prediction[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
