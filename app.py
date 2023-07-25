from flask import Flask,render_template,url_for,request
from flask_material import Material
import pandas as pd
import numpy as np
import sklearn.externals
import joblib
import sklearn.neighbors
import sklearn.svm
#import SVC

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("index.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
    if request.method == 'POST': 
        petal_length = request.form['petal_length'] 
        sepal_length = request.form['sepal_length'] 
        petal_width = request.form['petal_width'] 
        sepal_width = request.form['sepal_width']
		#model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float
        sample_data = [sepal_length,sepal_width,petal_length,petal_width]
        clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)
        knn_model = joblib.load('data/svm_model_iris.pkl') 
        result_prediction = knn_model.predict(ex1)
    return render_template('index.html', 
        petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
        result_prediction=result_prediction)


if __name__ == '__main__':
	app.run(debug=True)
