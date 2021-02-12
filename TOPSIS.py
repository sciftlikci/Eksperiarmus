from sklearn.preprocessing import *
from Transformer import *
from utils import *
from tqdm import tqdm
import time


class TOPSIS:
    # the constructor
    def __init__(self, appctext, cluster_df=None, pytops_params_df=None, selected_cluster=None, overall=False):
        self.appctext = appctext
        self.city_distance_df = None
        self.file_id = None
        self.transformer = None
        self.topsis_df = None
        self.cluster_df = cluster_df
        self.pytops_params_df = pytops_params_df

        self.data_pt_threshold = 0 if pytops_params_df is None else self.pytops_params_df["data_pt_threshold"].values[0]
        self.selected_cluster = sorted(self.cluster_df["ACCIDENT TYPE"].unique()) if pytops_params_df is None else [
            int(selected_cluster)]
        self.selected_criteria = self.pytops_params_df["selected_criteria"].values[
            0] if pytops_params_df is not None else ['File Closure Cost / Vehicle Value', 'File Closure Time',
                                                     "Percentage of Disclaim", "Percentage of Declines"]
        self.attributes = self.pytops_params_df["attributes"].values[0] if pytops_params_df is not None else [0, 0, 0,
                                                                                                              0]

        self.number_of_simulations = 150
        self.criterion_converter = {'File Closure Cost / Vehicle Value': "FILE / INSURANCE_TYPE_2",
                                    'File Closure Time': "FILE CLOSURE TIME", 'Percentage of Disclaim': 'PERCENTAGE OF DISCLAIM',
                                    'Percentage of Declines': 'PERCENTAGE OF DECLINES'}

        #  weights & variations
        self.weights = {'FILE / INSURANCE_TYPE_2': (0.062, 0.012), "FILE CLOSURE TIME": (0.077, 0.017),
                        'PERCENTAGE OF DISCLAIM': (0.254, 0.069), 'PERCENTAGE OF DECLINES': (0.218, 0.058),
                        'CUSTOMER SATISFACTION': (0.278, 0.09), 'TECHNICAL SERVICE SATISFACTION': (0.111, 0.033)}
        self.overall = overall

    def pbar_update(self, kind):
        time.sleep(1)
        kind_dict = {
            "extra_big": 40,
            "big": 20,
            "medium": 10,
            "small": 5,
            "extra_small": 1
        }
        self.pbar.update(kind_dict[kind])

    def get_cluster_label(self):
        if not self.overall:
            temp = set(self.cluster_df[self.cluster_df["ACCIDENT TYPE"].isin(self.selected_cluster)]["FILE ID"])
            temp2 = set(self.topsis_df["FILE ID"]).intersection(temp)
            self.adjuster_df = self.topsis_df[self.topsis_df["FILE ID"].isin(temp2)]
        else:
            self.topsis_df["ACCIDENT TYPE"] = self.cluster_df.set_index("FILE ID").loc[
                self.topsis_df["FILE ID"].values, "ACCIDENT TYPE"].values
            self.adjuster_df = self.topsis_df

    def prepare_selected_df(self):
        """
        Prepare dataframe for TOPSIS study
        """

        # get cluster label
        self.get_cluster_label()

        if self.topsis_df is None:
            return

        # Get adjuster's cities
        self.adjuster_cities = self.adjuster_df.groupby('ADJUSTER')['SERVICE CITY'].unique()

        # Get all adjusters in selected cluster
        self.all_adjusters = set(self.adjuster_df["ADJUSTER"].unique())

        if "File Closure Time" in self.selected_criteria:
            self.adjuster_df["FILE CLOSURE TIME"] = minmax_scale(self.adjuster_df["FILE CLOSURE TIME"])

        self.selected_criteria = [self.criterion_converter[criterion] for criterion in self.selected_criteria]
        self.adjuster_df = self.adjuster_df[self.selected_criteria + ["ADJUSTER"]] if not self.overall else self.adjuster_df[
            self.selected_criteria + ["ADJUSTER", "ACCIDENT TYPE"]]

    def find_nearest_cities_with_adjuster(self, results_df, accident_city):
        if accident_city is not None:
            nearest_cities = list(self.city_distance_df.nsmallest(3, accident_city)["CITY NAME"]) + accident_city.split(
                ', ')
            results_df["Available"] = ['✓' if bool(
                set(self.topsis_df[self.topsis_df["ADJUSTER"] == idx]["SERVICE CITY"]).intersection(nearest_cities)) else ''
                                         for
                                         idx in results_df.index]

        return results_df

    def desirability_mean(self, criterion):
        func = np.vectorize(desirability_mean_downstream)

        if criterion != "PERCENTAGE OF DISCLAIM":
            bottom = self.adjuster_df_mean[criterion].min() * 0.9
            top = self.adjuster_df_mean[criterion].max() * 1.1
            self.adjuster_df_mean[criterion] = func(self.adjuster_df_mean[criterion],
                                                  bottom,
                                                  top,
                                                  True)
        else:
            top = self.adjuster_df_mean[criterion].max() * 1.1
            self.adjuster_df_mean[criterion] = func(self.adjuster_df_mean[criterion],
                                                  top,
                                                  0,
                                                  False)

    def desirability_std(self, criterion):
        func = np.vectorize(desirability_stb_downstream)
        top = self.adjuster_df_mean[criterion].max() * 1.1

        adjuster_df_std_dict = {
            "FILE / INSURANCE_TYPE_2": top,
            "FILE CLOSURE TIME": top,
            "PERCENTAGE OF DECLINES": 0.5,
            "PERCENTAGE OF DISCLAIM": 0.1
        }

        self.adjuster_df_std[criterion] = func(self.adjuster_df_std[criterion],
                                             0,
                                             adjuster_df_std_dict[criterion])

    def calculate_desirability_exponent(self):
        self.adjuster_df = pd.DataFrame(index=self.adjuster_df_mean.index)
        func = np.vectorize(desirability_exponent_downstream)
        for criterion in self.selected_criteria:
            self.adjuster_df[criterion] = func(self.adjuster_df_mean[criterion],
                                             self.adjuster_df_std[criterion])

    def prepare_desirability(self):

        if "ACCIDENT TYPE" not in self.adjuster_df.columns:

            # Selecting adjusters above the data point threshold
            if self.data_pt_threshold < 2:
                self.adjuster_df_mean = self.adjuster_df.groupby("ADJUSTER").mean()
                self.pbar_update("medium")  # 20

                self.adjuster_df_std = pd.DataFrame(0, index=self.adjuster_df_mean.index,
                                                  columns=self.adjuster_df_mean.columns)
                m = self.adjuster_df.groupby("ADJUSTER").size() >= 2
                self.adjuster_df_std = self.adjuster_df_std.where(~m, self.adjuster_df.groupby("ADJUSTER")[
                    self.adjuster_df.columns].std())
                self.pbar_update("medium")  # 30

            else:
                self.adjuster_df = self.adjuster_df.groupby("ADJUSTER").filter(lambda x: len(x) > self.data_pt_threshold)
                self.pbar_update("small")  # 15

                self.adjuster_df_mean = self.adjuster_df.groupby("ADJUSTER").mean()
                self.pbar_update("small")  # 20

                self.adjuster_df_std = self.adjuster_df.groupby("ADJUSTER").std()
                self.pbar_update("medium")  # 30

        else:
            self.adjuster_df_mean = self.adjuster_df.groupby(["ACCIDENT TYPE", "ADJUSTER"]).mean()
            self.adjuster_df_std = pd.DataFrame(0, index=self.adjuster_df_mean.index,
                                              columns=self.adjuster_df_mean.columns)
            m = self.adjuster_df.groupby(["ACCIDENT TYPE", "ADJUSTER"]).size() >= 2
            self.adjuster_df_std = self.adjuster_df_std.where(~m, self.adjuster_df.groupby("ADJUSTER")[
                self.adjuster_df_mean.columns].std())
            self.adjuster_df_std.fillna(0, inplace=True)

        for col in self.adjuster_df_mean.columns:
            self.desirability_mean(col)
            self.desirability_std(col)
        self.pbar_update("medium")  # 40 # 30

        self.calculate_desirability_exponent()
        self.pbar_update("medium")  # 50 # 40

        if self.overall:
            sorted_df_dict = {}
            self.selected_cluster = sorted(set([self.adjuster_df.index[i][0] for i in range(len(self.adjuster_df.index))]))

            for col in self.selected_criteria:
                # step 5
                sorted_df_dict[col] = self.adjuster_df.sort_values(["ACCIDENT TYPE"] + [col])[col]

                # step 6
                for cluster in self.selected_cluster:
                    sorted_df_dict[col].loc[[cluster]] = sorted_df_dict[col].loc[[cluster]].rank(ascending=True,
                                                                                                 method="min")
                sorted_df_dict[col] = sorted_df_dict[col].div(
                    sorted_df_dict[col].reset_index().groupby(["ACCIDENT TYPE"]).count()["ADJUSTER"])
                self.pbar_update("extra_small")  # 41

                # step 7
                sorted_df_dict[col] = sorted_df_dict[col].reset_index()
                p_e_c_k = sorted_df_dict[col][0].values.flatten()
                n_e_c = sorted_df_dict[col].groupby(["ACCIDENT TYPE", "ADJUSTER"]).transform("count").values.flatten()
                sum_n_e_c = sorted_df_dict[col].groupby(["ADJUSTER"]).transform("count")[
                    "ACCIDENT TYPE"].values.flatten()
                sorted_df_dict[col]["p_e_c_k"] = np.divide(np.multiply(p_e_c_k, n_e_c), sum_n_e_c)
                sorted_df_dict[col]["p_e_k"] = sorted_df_dict[col].groupby(["ADJUSTER"])["p_e_c_k"].transform(
                    "sum")
                self.pbar_update("extra_small")  # 42

                # step 8
                p_e_c_k_minus_p_e_k = (sorted_df_dict[col]["p_e_c_k"] - sorted_df_dict[col]["p_e_k"]) ** 2
                self.pbar_update("extra_small")  # 43
                sorted_df_dict[col]["s_p_e_c_k"] = np.divide(np.multiply(p_e_c_k_minus_p_e_k, n_e_c), sum_n_e_c)
                self.pbar_update("extra_small")  # 44
                sorted_df_dict[col]["s_p_e_k"] = sorted_df_dict[col].groupby(["ADJUSTER"])["s_p_e_c_k"].transform(
                    "sum")

                # step 9 - final desirability
                func1 = np.vectorize(desirability_mean_downstream)
                func2 = np.vectorize(desirability_stb_downstream)
                func3 = np.vectorize(desirability_exponent_downstream)

                sorted_df_dict[col]["p_e_k"] = func1(sorted_df_dict[col]["p_e_k"],
                                                     sorted_df_dict[col]["p_e_k"].max(),
                                                     sorted_df_dict[col]["p_e_k"].min(), False)

                sorted_df_dict[col]["s_p_e_k"] = func2(sorted_df_dict[col]["s_p_e_k"], 0,
                                                       sorted_df_dict[col]["s_p_e_k"].max())
                sorted_df_dict[col]["D"] = func3(sorted_df_dict[col]["p_e_k"],
                                                 sorted_df_dict[col]["s_p_e_k"])
                self.pbar_update("extra_small")  # 45

            self.adjuster_df = pd.DataFrame(0, index=sorted_df_dict[self.selected_criteria[0]]["ADJUSTER"].unique(),
                                          columns=self.selected_criteria)

            for col in self.selected_criteria:
                self.adjuster_df.loc[sorted_df_dict[col]["ADJUSTER"], col] = sorted_df_dict[col]["D"].values

    def topsis_analysis(self, accident_city=None):

        self.pbar = tqdm(total=100, position=0, leave=True,
                         desc="Adjusters are being ranked..." if accident_city is not None else "Adjuster overall rankings are being prepared...")

        # prepare dataframe for topsis
        self.prepare_selected_df()
        self.pbar_update("medium")  # 10

        if self.topsis_df is None:
            return None, None

        # Desirability
        self.prepare_desirability()

        # Tracking eliminated adjusters from study
        eliminated_adjusters = self.all_adjusters ^ set(self.adjuster_df.index)

        # Normalizing the Decision Matrix - adjuster dataframe
        b = normalize(self.adjuster_df.values, norm='l2', axis=0)
        self.pbar_update("small")  # 45 # 55

        # Importing the Weights
        current_weights = [self.weights[weight] for weight in self.selected_criteria]

        # Creating lower and upper bounds for weights
        min1 = [max(val[0] * (1 - val[1]), 0) for val in current_weights]
        max1 = [min(val[0] * (1 + val[1]), 1) for val in current_weights]

        # Create empty dataframe for saving simulation results
        sim_df = pd.DataFrame(columns=self.adjuster_df.index)
        interval_df = pd.DataFrame(columns=self.adjuster_df.index)
        order_df = pd.DataFrame(columns=self.adjuster_df.index)
        self.pbar_update("small")  # 60 # 60

        # Simulation cycles
        for j in range(0, self.number_of_simulations):
            # Drawing criteria weights from random distribution
            y1 = [np.random.uniform(min1[i], max1[i], size=1)[0] for i in range(len(current_weights))]
            c = y1 / np.sum(y1)
            d = c * b

            # Getting mean relative closeness scores
            array = np.array(upstream(midstream(d, downstream(d, self.attributes))))

            # Saving simulation cycle results
            sim_df.loc[j, :] = array
            order_df.loc[j, :] = array.argsort()[::-1]
        self.pbar_update("medium")  # 70 # 70

        mean_of_array = np.array([round(100 * num, 2) for num in sim_df.mean().values])

        results_df = pd.DataFrame()
        results_df["ADJUSTER"] = self.adjuster_df.index
        results_df["Mean Score"] = mean_of_array
        for col in sim_df.columns:
            interval_df.loc[0, col] = mean_confidence_interval(sim_df[col])

        results_df = results_df.groupby(["ADJUSTER"]).mean()
        self.pbar_update("medium")  # 80 #80

        results_df["City"] = [', '.join(self.adjuster_cities.loc[adjuster]) for adjuster in results_df.index]
        results_df.sort_values(by="Mean Score", axis=0, ascending=False, inplace=True)
        self.pbar_update("medium")  # 90 # 90
        results_df["Ranking"] = np.arange(1, len(results_df.index) + 1)
        results_df = self.find_nearest_cities_with_adjuster(results_df,
                                                          accident_city) if accident_city is not None else results_df
        results_df["Confidence Interval"] = [interval_df.at[0, idx] for idx in results_df.index]
        self.pbar_update("medium")  # 100 # 100
        self.pbar.close()
        return results_df, eliminated_adjusters
