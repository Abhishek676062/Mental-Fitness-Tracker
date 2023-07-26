from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load your trained regression model
model = pickle.load(open("model.sav", "rb"))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input features from the form
             inputQuery1 = request.form['query1']
             inputQuery2 = request.form['query2']
             inputQuery3 = request.form['query3']
             inputQuery4 = request.form['query4']
             inputQuery5 = request.form['query5']
             inputQuery6 = request.form['query6']
             inputQuery7 = request.form['query7']
             inputQuery8 = request.form['query8']
             inputQuery9 = request.form['query9']


            # Add more features as needed
             l = LabelEncoder()
             Country_encoded = l.fit_transform([inputQuery1])
            # Perform prediction using the loaded model
             prediction = model.predict([[Country_encoded[0], inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9]])[0]

             return render_template('result.html', prediction=prediction)
        except Exception as e:
            error_message = str(e)
            return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
