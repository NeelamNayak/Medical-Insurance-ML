
import numpy as np
import pandas as pd
import pickle
import json
import warnings 
import config
warnings.filterwarnings("ignore")

class MedicalInsurance:
    def __init__(self, age, sex, bmi, children, smoker, region):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

        self.sex = "sex_" + sex
        self.smoker = "smoker_" + smoker
        self.region = "region_" + region

    def load_models(self):
        with open("/Machine Learning/Decision Tree CR/36_Neelam Thakur_Medical insurance Decision Tree_e2e Model/project_app/Linear_Model.pkl","rb") as f:
            self.model = pickle.load(f)

        with open("/Machine Learning/Decision Tree CR/36_Neelam Thakur_Medical insurance Decision Tree_e2e Model/project_app/Medical_Insurance_data.json","r") as f:
            self.json_data = json.load(f)

    def get_predicted_charges(self):
        
        self.load_models()

        region_index = region_index = list(self.json_data['columns']).index(self.region)
        sex_index = sex_index = list(self.json_data['columns']).index(self.sex)
        smoker_index = smoker_index = list(self.json_data['columns']).index(self.smoker)
        train_acc =  self.json_data['Training Accuracy']
        test_acc = self.json_data['Testing Accuracy']
        test_array = np.zeros(len(self.json_data['columns']))


        test_array[0] = self.age
        test_array[1] = self.bmi
        test_array[2] = self.children
        test_array[sex_index] = 1
        test_array[smoker_index] = 1
        test_array[region_index] = 1

        print("Test Array --->\n", test_array)
 
    

        charges = round(self.model.predict([test_array])[0],2)
        return charges, train_acc, test_acc

if __name__ == "__main__":
    age = 40
    sex = "male"
    bmi = 27
    children = 2
    smoker = "yes"
    region = "northwest"

 

    
    med_inst = MedicalInsurance(age, sex, bmi, children, smoker, region)
    charges,train_acc,test_acc = med_inst.get_predicted_charges()
    print(f"Predicted Medical Insurance Charges is :{charges} /-Rs.") 
    print(f"Traning Accuracy: {train_acc}   Testing Accuracy: {test_acc}") 
    