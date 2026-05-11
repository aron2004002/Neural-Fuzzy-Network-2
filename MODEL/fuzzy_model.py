import numpy as np


class NeuroFuzzySystem:

    def __init__(self):
        pass

    def fuzzify_glucose(self, glucose):

        if glucose < 100:
            return 0.2

        elif 100 <= glucose < 140:
            return 0.5

        else:
            return 0.9

    def fuzzify_bmi(self, bmi):

        if bmi < 18:
            return 0.2

        elif 18 <= bmi < 30:
            return 0.5

        else:
            return 0.9

    def predict_risk(self, glucose, bmi):

        glucose_score = self.fuzzify_glucose(glucose)
        bmi_score = self.fuzzify_bmi(bmi)

        risk = (glucose_score + bmi_score) / 2

        if risk > 0.6:
            return "High Diabetes Risk"

        else:
            return "Low Diabetes Risk"