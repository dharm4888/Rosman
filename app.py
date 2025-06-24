from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Use the full absolute path to your saved model
model_path = r"C:\Users\91878\AI COURSE DIGICROME\rosman sale\saved_models\rf_pipeline_2025-06-23-22-06-53.pkl"
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_order = [
            'Store', 'Promo', 'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'Promo2', 'Month', 'DayOfWeek', 'IsWeekend'
        ]

        input_values = [float(request.form.get(feature)) for feature in feature_order]
        input_df = pd.DataFrame([input_values], columns=feature_order)
        prediction = model.predict(input_df)[0]

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
