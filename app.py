from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('solar_power_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render the input form

@app.route('/home1')
def home1():
    return render_template('home1.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        # Get input values from the form
        daily_yield = float(request.form['daily_yield'])
        total_yield = float(request.form['total_yield'])
        ambient_temp = float(request.form['ambient_temp'])
        module_temp = float(request.form['module_temp'])
        irradiation = float(request.form['irradiation'])
        hour = float(request.form['hour'])
        min = float(request.form['min'])
        day = float(request.form['day'])
        month = float(request.form['month'])
        year = float(request.form['year'])
        dow = float(request.form['dow'])

        # Combine inputs into a feature array
        features = np.array([[daily_yield, total_yield, ambient_temp, module_temp, irradiation, hour, min, day, month, year, dow]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('result1.html', prediction=prediction)

    except ValueError as e:
        return f"Invalid input: {e}", 400
    
@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    # Load the CSV file into a DataFrame
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    # Ensure the file has the required columns
    required_columns = ['DAILY_YIELD', 'TOTAL_YIELD',
                        'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                        'hour', 'minute','day', 'month', 'year', 'day_of_week']
    
    if not all(col in data.columns for col in required_columns):
        return f"File is missing required columns: {required_columns}", 400

    # Make predictions
    features = data[required_columns]
    predictions = model.predict(features)

    # Add predictions to the DataFrame
    data['Predicted_AC_Power'] = predictions

    # Convert DataFrame to HTML for display
    results_html = data.to_html(classes='table table-striped')

    return render_template('result2.html', table=results_html)

if __name__ == '__main__':
    app.run(debug=True)
