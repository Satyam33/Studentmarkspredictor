import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib

app = Flask(__name__)

model = joblib.load("student_mark_predictor.pkl")

df = pd.DataFrame()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global  df

    input_feature = [int(x) for x in request.form.values()]
    feature_values = np.array(input_feature)

    if input_feature[0] < 0 or input_feature[0]>12:
        return  render_template('index.html',prediction_text = 'please enter a valid hours between 1 to 12')

    output = model.predict([feature_values])[0][0].round(2)

    df = pd.concat([df,pd.DataFrame({'study Hours':input_feature,'Perdict Output':[output]})],ignore_index=True)

    print(df)
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html',prediction_text='You will get [{}%] marks,when  you do study [{}] hours per day '.format(output, int(feature_values[0])))

if __name__ == '__main__':
    app.run()


