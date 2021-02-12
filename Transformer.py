from PyQt5.QtWidgets import QMessageBox
from utils import listOfTuples, text_similarity
from Errors import *
import os
import pandas as pd
import numpy as np


def get_info_message_text_by_code_transformer(message_code):
    msg = QMessageBox()
    msg.setWindowTitle("Adjuster Evaluation System")

    switcher = {
        1: "There is a lack of data. Please check the relevant accident file.",
        2: "No such Insurance Type was found. Please check the relevant accident file.",
        3: "No such Service Type was found. Please check the relevant accident file.",
        4: "No such City was found. Please check the relevant accident file.",
        5: "No such Brand found. Please check the relevant file.",
    }
    msg.setText(switcher.get(message_code, "Error! Operation failed."))
    msg.setIcon(QMessageBox.Critical)
    msg.exec_()


class DataTransformer(object):
    """Make data transformation"""

    def __init__(self, appctext):
        self.appctext = appctext
        self.city_df = None
        self.price_df = None
        self.transformed_df = pd.DataFrame(
            columns=["INSURANCE_TYPE", "SERVICE_TYPE", "SERVICE CITY", "ESTIMATED DAMAGE COST", "VEHICLE_VALUE"])

    def insurance_type_transform(self):
        # trafik type_1, kasko type_2 olarak belirtildi.
        if not set(list(self.input_data["INSURANCE_TYPE"].unique())) <= {"INSURANCE_TYPE_1", "INSURANCE_TYPE_2"}:
            get_info_message_text_by_code_transformer(2)
            raise InsuranceTypeError
        else:
            self.transformed_df["INSURANCE_TYPE"] = [0 if tp.lower() == "trafik" else 1 for tp in
                                                             list(self.input_data["INSURANCE_TYPE"])]

    def service_type_transform(self):
        if not set(list(self.input_data["SERVICE_TYPE"].unique())) <= {"SPECIAL", "AUTHORIZED"}:
            get_info_message_text_by_code_transformer(3)
            raise ServiceTypeError
        else:
            self.transformed_df["SERVICE_TYPE"] = [0 if tp.lower() == "special" else 1 for tp in
                                                  list(self.input_data["SERVICE_TYPE"])]

    def city_sei_transform(self):
        if not set(list(self.input_data["SERVICE CITY"].unique())) <= set(list(self.city_df["City"].unique())):
            get_info_message_text_by_code_transformer(4)
            raise CityError
        else:
            self.transformed_df["SERVICE CITY"] = [self.city_df[self.city_df["City"] == city]['SosyoEko'].values[0] for
                                                 city in list(self.input_data["SERVICE CITY"])]

    def insurance_value_transform_downstream(self, car, model):
        temp = self.price_df[(self.price_df["BRAND_NAME"] == car) & (self.price_df["MODEL_NAME"] == model)]

        if len(temp) == 0:
            similarity_score = {model2: text_similarity(model, model2) for model2 in
                                self.price_df[self.price_df["BRAND_NAME"] == car]["MODEL_NAME"]}
            similar_model = max(similarity_score, key=similarity_score.get)
            temp = self.price_df[(self.price_df["BRAND_NAME"] == car) & (self.price_df["MODEL_NAME"] == similar_model)]
            price_ratio = (temp["ORIGINAL VEHICLE VALUE"].values[0], temp["VEHICLE_VALUE"].values[0])
        else:
            price_ratio = (temp["ORIGINAL VEHICLE VALUE"].values[0], temp["VEHICLE_VALUE"].values[0])

        return price_ratio

    def insurance_value_transform(self):
        if not set(list(self.input_data["BRAND_NAME"].unique())) <= set(list(self.price_df["BRAND_NAME"].unique())):
            get_info_message_text_by_code_transformer(5)
            raise CarBrandError
        else:
            price_tuple = listOfTuples(list(self.input_data["BRAND_NAME"]), list(self.input_data["MODEL_NAME"]))
            insurance_value_transformed = [self.insurance_value_transform_downstream(element[0], element[1]) for
                                           element in price_tuple]
            self.transformed_df["ORIGINAL VEHICLE VALUE"], self.transformed_df["VEHICLE_VALUE"] = zip(
                *insurance_value_transformed)

    def insurance_value_transform_outstream(self, input_df):
        price_tuple = listOfTuples(list(input_df["BRAND_NAME"]), list(input_df["MODEL_NAME"]))
        func = np.vectorize(self.insurance_value_transform_downstream)
        insurance_value_transformed = np.array([func(element[0], element[1])[0] for
                                                element in price_tuple])
        return np.divide(input_df["TOTAL_FILE_COST"].values, insurance_value_transformed)

    def damage_level_transform_downstream(self, predicted_damage_level, value):
        return 0 if 0 <= predicted_damage_level / value <= 0.1 else 0.5 if 0.1 < predicted_damage_level / value <= 0.3 else 1

    def damage_level_transform(self):
        self.damage_tuple = listOfTuples(list(self.input_data["ESTIMATED DAMAGE COST"].values),
                                         list(self.transformed_df["ORIGINAL VEHICLE VALUE"]))
        self.transformed_df["ESTIMATED DAMAGE COST"] = [self.damage_level_transform_downstream(element[0], element[1])
                                                       for element in self.damage_tuple]

    def transform(self, input_data, cluster=False):
        try:
            if not isinstance(input_data, pd.DataFrame):
                self.input_data = pd.DataFrame([input_data],
                                               columns=["INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST",
                                                        "SERVICE CITY", "BRAND_NAME", "MODEL_NAME"])
            else:
                self.input_data = input_data [["INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST",
                                              "SERVICE CITY", "BRAND_NAME", "MODEL_NAME"]] if not cluster else \
                                  input_data[["FILE ID", "INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST",
                                              "SERVICE CITY", "VEHICLE_VALUE", 'ORIGINAL VEHICLE VALUE']]
        except Exception as e:
            get_info_message_text_by_code_transformer(1)
            return

        try:
            self.insurance_type_transform()
            self.service_type_transform()
            self.city_sei_transform()
            if not cluster:
                self.insurance_value_transform()
                self.damage_level_transform()
            else:
                self.transformed_df.insert(0, "FILE ID", self.input_data["FILE ID"])
                self.transformed_df["ESTIMATED DAMAGE COST"] = self.input_data["ESTIMATED DAMAGE COST"]
                self.transformed_df["VEHICLE_VALUE"] = self.input_data["VEHICLE_VALUE"]
                self.transformed_df['ORIGINAL VEHICLE VALUE'] = self.input_data['ORIGINAL VEHICLE VALUE']
            return self.transformed_df
        except (InsuranceTypeError, ServiceTypeError, CityError, CarBrandError):
            return None
