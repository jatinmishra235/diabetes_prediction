from flask import Flask, jsonify, request, render_template
import logging
import pandas as pd
import pickle
import os
from flask_cors import cross_origin

logging.basicConfig(filename='diabetes.log', level=logging.INFO)

app = Flask(__name__)

current_script_path = os.path.abspath(__file__)
app_root = os.path.dirname(current_script_path)

print(current_script_path)
print(app_root)

@app.route("/", methods = ["GET"])
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route("/predict", methods= ["GET","POST"])
@cross_origin()
def predictDiabetes():
    if request.method == "POST":
        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        SkinThickness = request.form.get('SkinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')

        df = pd.DataFrame({'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age}, index = [0])

        logging.info("got all data..")

        model_path = os.path.join(app_root, 'models', 'diabetes_logistic_reg.pkl')

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        pred = model.predict(df)

        logging.info('got predictions!!')

        if pred == 0:
            message = "Congratulations!! you don't have diabetes"
        else:
            message = "Oops, you have Diabetes"
        
        logging.info(f"prediction is {message}")

        return render_template('result.html', message = message)





if __name__=="__main__":
    app.run()