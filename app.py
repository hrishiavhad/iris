from flask import Flask ,render_template,request,jsonify
import pickle

app= Flask(__name__)

model=pickle.load(open('lr_model.pkl','rb'))

@app.route('/')
def home():
    result= ''
    return render_template('index.html' , **locals())

@app.route('/predict' , methods=['POST','GET'])
def predict():
    Sepal_Length=float(request.form['Sepal_Length'])
    Sepal_Width=float(request.form['Sepal_Width'])
    Petal_Length=float(request.form['Petal_Length'])
    Petal_Width=float(request.form['Petal_Width'])
    result=model.predict([[Sepal_Length,Sepal_Width,Petal_Length,Petal_Width]])[0]

    if result==0:
        return 'Iris-Sentosa'

    elif result==1:
        return 'Iris-versicolor'

    else :
        return 'Iris-virginica'
   
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=1800)