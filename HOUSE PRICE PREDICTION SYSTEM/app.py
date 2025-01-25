from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load your model and data
df = pd.read_csv('new.csv')
x = df[['location_encoded', 'Total_Area', 'bedrooms', 'baths']]
y = df['price']

# Initialize model and encoder
lr = LinearRegression()
le = LabelEncoder()

# Train the model
lr.fit(x, y)

# Define a route for the home page (rendering the form)
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        location = request.form['location_encoded']
        total_area = float(request.form['total_area'])
        bedrooms = int(request.form['bedrooms'])
        baths = int(request.form['baths'])

        # Encode location
        location_encoded = le.fit_transform([location])[0]

        # Prepare input for prediction
        input_data = np.array([[location_encoded, total_area, bedrooms, baths]])

        # Predict price
        predicted_price = int (lr.predict(input_data)[0])

        # Return the predicted price to the frontend
        return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
