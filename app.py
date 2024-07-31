from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the data
df = pd.read_csv('distributor_consumption.csv')
df.drop(columns=['order_place_date'], inplace=True)

# One-hot encode categorical variables without converting to int
df = pd.get_dummies(df, columns=['distributor_name'])

# Define the columns for the input data
X_columns = df.drop(columns=['depletion_time']).columns

X = df.drop(columns=['depletion_time'])
y = df['depletion_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create DataFrame for evaluation
results_df = pd.DataFrame({
    'Actual_Depletion_Time': y_test,
    'Predicted_Depletion_Time': y_pred
})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    distributor_name = request.form['distributor_name']
    order_place_date = request.form['order_place_date']
    lot_size = int(request.form['lot_size'])

    # Prepare the input data
    new_request = pd.DataFrame({
        'distributor_name': [distributor_name],
        'order_place_date': [order_place_date],
        'lot_size': [lot_size]
    })

    new_request['order_place_date'] = pd.to_datetime(new_request['order_place_date'])
    new_request_preprocess = pd.get_dummies(new_request, columns=['distributor_name'])
    new_request_preprocess = new_request_preprocess.reindex(columns=X_columns, fill_value=0)

    # Predict the depletion time
    predicted_time = model.predict(new_request_preprocess)[0].astype(int)
    estimated_depletion_date = new_request['order_place_date'][0] + pd.Timedelta(days=predicted_time)

    return render_template('result.html', order_place_date=new_request['order_place_date'][0].strftime('%Y-%m-%d'), lot_size=lot_size, predicted_time=predicted_time, estimated_depletion_date=estimated_depletion_date.strftime('%Y-%m-%d'))


@app.route('/evaluation')
def evaluation():
    # Convert the DataFrame to a list of dictionaries for rendering
    results_list = results_df.head(10).to_dict(orient='records')
    return render_template('evaluation.html', results=results_list)

if __name__ == '__main__':
    app.run(debug=True)

