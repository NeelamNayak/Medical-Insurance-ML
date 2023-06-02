from flask import Flask, jsonify, render_template, request
from project_app.utils import MedicalInsurance

app = Flask(__name__)

@app.route("/")
def hello_flask():
    print("Welcome to Medical Insurance System")
    return render_template("index.html")

@app.route("/predict_charges",methods = ['POST','GET'])
def get_insurance_charges():

    if request.method == 'GET':
        print("We are in GET method")

    # data = request.form
    # print("Data \n",data)

        age = int(request.args.get("age"))
        sex = request.args.get("sex")
        bmi = float(request.args.get("bmi"))
        children = int(request.args.get("children"))
        smoker = request.args.get("smoker")
        region = request.args.get("region")
    
        med_inst = MedicalInsurance(age, sex, bmi, children, smoker, region)
        charges, train_acc, test_acc = med_inst.get_predicted_charges()
        return render_template("index.html",prediction = charges, train = train_acc, test = test_acc)

app.run(debug = True)