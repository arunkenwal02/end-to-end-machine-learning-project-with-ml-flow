from django.shortcuts import render
import pickle
from rest_framework.generics import GenericAPIView
import pandas as pd
from rest_framework.response import Response




class Predict(GenericAPIView):
    def post(self, request):
        data = {
            'person_age': request['person_age'],
            'person_income': request['person_income'],
            'person_home_ownership': request['person_home_ownership'],
            'person_emp_length': request['person_emp_length'],
            'loan_intent': request['loan_intent'],
            'loan_amnt': request['loan_amnt'],
            'loan_int_rate': request['loan_int_rate'],
            'loan_percent_income': request['loan_percent_income'],
            'cb_person_default_on_file': request['cb_person_default_on_file'],
            'cb_person_cred_hist_length': request['cb_person_cred_hist_length']
        }
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict(pd.DataFrame(data))
        return Response(prediction)

