import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from flask import Flask, request, render_template
app = Flask(__name__)
models = joblib.load("Covid_19_case_predictor_model_10.pkl")
df = pd.DataFrame()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET'])
def predict():
    global df
    input_features = [int(x) for x in request.form.values()]
    s = [459]
    lst3 = sum(s+input_features)
    Amit1 = np.array([[lst3]])
    from sklearn.preprocessing import PolynomialFeatures
    polyFeat = PolynomialFeatures(degree=3)

    Abc = polyFeat.fit_transform(Amit1)
    #import numpy as np

    features_value=np.array(Abc)


    #validate input days
    if input_features[0] < 0 :
        return render_template('index.html',prediction_text='please inter valid number of days')

    outputs = (round(int(models.predict( features_value ))/1000000,2),'Million')

    # input and predicted value store in df then save in csv file
    df=pd.concat([df,pd.DataFrame({'id': input_features,'Predicted Output':[outputs]})],ignore_index=True)
    print(df)
    df.to_csv('smp_data_form_app.csv')

    return render_template('index.html', prediction_text='the total number of cases [{}] after [{}] days '.format(outputs,lst3))
if __name__ == "__main__":
    app.run(debug=True)





