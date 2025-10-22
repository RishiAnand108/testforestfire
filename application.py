import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import sklearn.preprocessing as StandardScaler

application = Flask(__name__) 
app=application

## import ridge regressor and standard scaler
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    # Serve the main form at root for convenience
    return render_template('home.html')


@app.route('/home')
def home_alias():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Parse and validate inputs
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
        except (TypeError, ValueError) as e:
            # Return the form with an error message if parsing fails
            return render_template('home.html', result=None, error=f"Invalid input: {e}")

        try:
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = float(ridge_model.predict(new_data_scaled)[0])
        except Exception as e:
            return render_template('home.html', result=None, error=f"Prediction error: {e}")

        # Return the template with the computed result
        return render_template('home.html', result=result)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')