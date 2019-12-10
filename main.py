from flask import Flask, request, jsonify, render_template,redirect
from flask_sqlalchemy import SQLAlchemy
import time

#create flask app
app = Flask(__name__)
#create database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///irisFlowerCollection.db'
db = SQLAlchemy(app)

#---------------------MODEL---------------------#
import IrisModel

clf, classes = IrisModel.irisFlowerModel()

#---------------------DATABASE CLASSES---------------------#
class FlowerInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sepal_length = db.Column(db.Integer)
    sepal_width = db.Column(db.Integer)
    petal_length = db.Column(db.Integer)
    petal_width = db.Column(db.Integer)
    species = db.Column(db.String(80))

#---------------------API ROUTES---------------------#

#home page
@app.route("/") 
def home(): 
    return render_template("home.html")

#prediction
def makePrediction(sepal_length,sepal_width,petal_length,petal_width):

    import numpy as np

    item = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=float).reshape(1,-1)
    score = clf.predict(item)
    print(score)
    results = (classes[score[0] == 1])[0]
    
    return results
    
@app.route('/predict', methods=['GET','POST'])
def predict():

    #handle post request
    if request.method == "POST":
        #read in data input from user
        
            sepal_length = request.form['sepal_length']
            sepal_width = request.form['sepal_width']
            petal_length = request.form['petal_length']
            petal_width = request.form['petal_width']

            #check for missing data
            missing = False
            for key, value in request.form.items():
                if value == "":
                    missing = True
            
            if missing:
                #add message to indicate missing information and to redo
                missingInformation = f"Missing information, redo entry"
                return render_template("predict.html", missingInformation=missingInformation)
            else:
                #if no information is missing continue processing

                #predict after parsing the data
                irisClass = makePrediction(sepal_length,sepal_width,petal_length,petal_width)

                #create new flower
                newFlower = FlowerInstance(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                species=irisClass)
     
                #add new flower to the database
                db.session.add(newFlower)
                # #commit changes to database
                db.session.commit()

                #redirect back to the same page 
                return redirect(request.url)

    #load page if no post request but have navigated to page
    return render_template("predict.html")
    


if __name__ == '__main__':

    #connect to server
    
    #specify the host and port to run the app on
    HOST = "127.0.0.1"
    PORT = 4000

    app.run(HOST,PORT, debug=True)


