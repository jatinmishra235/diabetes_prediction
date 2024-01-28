from flask import Flask, jsonify, request, render_template
import logging
import pandas as pd
import pickle

logging.basicConfig(filename='diabetes.log', level=logging.INFO)

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def homepage():
    return render_template("index.html")

@app.route("/predict", methods= ["GET","POST"])
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

        with open(r'C:\Users\user\Desktop\Diabetes_prediction\models\diabetes_logistic_reg.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        pred = model.predict(df)

        logging.info('got predictions!!')

        if pred == 0:
            messege = "Congratulations!! you don't have diabetes"
        else:
            messege = "Oops, you have Diabetes"
        
        logging.info(f"prediction is {messege}")

        return render_template('result.html', message = messege)





if __name__=="__main__":
    app.run()