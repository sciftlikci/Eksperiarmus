from pyclustering.cluster.kmedoids import kmedoids
import pickle
from joblib import dump, load
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate
from Transformer import *
from utils import *
import datetime
from PyQt5.QtWidgets import *
import numpy as np
from scipy.stats import mode
import time
from tqdm import tqdm


class ExtraTrees:
    # the constructor
    def __init__(self, appctext, data, transformer, classifier):
        self.appctext = appctext
        self.classifier = classifier
        self.data = data
        self.idx = len(self.data)
        self.transformer = transformer
        self.classifier_features = ["INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST", "SERVICE CITY",
                                    "VEHICLE_VALUE"]

    # the fit method
    def fit(self, x, y):
        """
        :param x: N X p dataframe
        :param y: N X 1 dataframe
        """
        cv_results = cross_validate(ExtraTreesClassifier(), x.values, y.values, cv=min(5, min(y.value_counts())),
                                    scoring='f1_macro',
                                    return_estimator=True,
                                    verbose=1)
        self.classifier = cv_results['estimator'][np.argmax(cv_results['test_score'])]
        dump(self.classifier, self.appctext.get_resource('best_classifier_f1_macro.joblib'))

    # method to predict
    def predict(self, x, file_id):
        x = self.transformer.transform(x)
        try:
            k_cluster = \
            mode([self.classifier.predict(x[self.classifier_features].values.reshape(1, -1))[0] for i in range(5)])[0][
                0]

            x["ACCIDENT TYPE"] = k_cluster
            try:
                x.insert(0, "FILE ID", file_id)
            except:
                x.at[0, "FILE ID"] = file_id
            if file_id not in self.data["FILE ID"].values:
                self.data.loc[self.idx, :] = list(x[self.data.columns].values[0])
                self.data.to_csv(self.appctext.get_resource('clustering.csv'), index=False, header=True)
            return k_cluster

        except TypeError:
            return


class KMedoidsClusterer:
    # the constructor
    def __init__(self, appctext, clustering_df, transformer):
        """
        :param max_depth: the maximum tree height/depth
        """
        self.appctext = appctext
        self.cv_threshold = 0.25
        self.max_cluster = 200
        self.clustering_df = clustering_df
        self.clustering_df_cols = {'FILE ID', "INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST",
                                   "SERVICE CITY",
                                   "BRAND_NAME", "MODEL_NAME"}
        self.clustering_features = ["INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST", "SERVICE CITY",
                                    "VEHICLE_VALUE"]
        self.missing_cols = ""
        self.transformer = transformer

    def pbar_update(self, kind):
        kind_dict = {
            "extra_big": 40,
            "big": 20,
            "medium": 10,
            "small": 5,
            "extra_small": 1,
            "xx_small": 0.25
        }
        self.pbar.update(kind_dict[kind])

    def get_info_message_text_by_code_clusterer(self, message_code):
        msg = QMessageBox()
        msg.setWindowTitle("Evaluation System")

        switcher = {
            1: ', '.join(self.missing_cols) + " columns are missing. Please check the related Excel file.",
        }
        msg.setText(switcher.get(message_code, "Error! Operation failed."))
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

        return

    def transform_data(self):
        self.clustering_df = self.transformer.transform(self.clustering_df, True)

    # the fit method
    def fit(self, clustering_df):
        """
        :param clustering_df:
        :param x: N X p numpy.ndarray
        """
        self.clustering_df = clustering_df
        self.pbar = tqdm(total=100, position=0, leave=True, desc="New accident types are being created ...")

        self.clustering_df.drop("ACCIDENT TYPE", inplace=True, axis=1)
        self.pbar_update("small")

        self.clustering_df.dropna(inplace=True)
        self.pbar_update("small")

        # Optimal k search
        global k
        for k in range(2, min(self.max_cluster, len(self.clustering_df))):
            # Train a kmedoids with best k
            kmedoids_instance = kmedoids(self.clustering_df[self.clustering_features].values,
                                         init_medoids(self.clustering_df[self.clustering_features].values,
                                                      k),
                                         ccore=True)

            # run cluster analysis and obtain results
            kmedoids_instance.process()
            self.pbar_update("xx_small")
            if self.coeff_variance_check(kmedoids_instance):
                self.pbar.n = min(self.max_cluster, len(self.clustering_df))
                self.pbar.last_print_n = min(self.max_cluster, len(self.clustering_df))
                break

        print("\nOptimal accident types are created, results are being recorded ...")
        self.save_results()
        return k

    # method to predict
    def predict(self, kmedoids_instance):
        """
        :param kmedoids_instance:
        :return: predicted_label
        """
        clusters = {i + 1: cluster_idx for i, cluster_idx in enumerate(kmedoids_instance.get_clusters())}
        for key, val in clusters.items():
            self.clustering_df.loc[val, "ACCIDENT TYPE"] = int(key)

    def coeff_variance_check_downstream(self, x):
        return 1 if x <= self.cv_threshold else 0

    def coeff_variance_check(self, kmedoids_instance):
        self.predict(kmedoids_instance)
        clustering_df_grouped = self.clustering_df.groupby(["ACCIDENT TYPE"]).std() / self.clustering_df.groupby(
            ["ACCIDENT TYPE"]).mean()
        func = np.vectorize(self.coeff_variance_check_downstream)
        masked = func(clustering_df_grouped["ORIGINAL VEHICLE VALUE"].values)

        return False if np.mean(masked) <= 0.8 else True

    def save_results(self):
        self.clustering_df.to_csv(self.appctext.get_resource('clustering.csv'), index=None)
        self.clustering_df.drop('FILE ID', axis=1, inplace=True)

        with open(self.appctext.get_resource('last_clustering_date.pickle'), 'wb') as handle:
            pickle.dump(datetime.datetime.now().strftime("%d/%m/%Y"), handle, protocol=pickle.HIGHEST_PROTOCOL)
