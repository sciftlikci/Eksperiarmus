from datetime import datetime
from math import pi
import jinja2
from docx.shared import Mm
from docx2pdf import convert
from docxtpl import DocxTemplate, InlineImage
from scipy import stats
from TOPSIS import *
import numpy as np
import datetime
import os
import pygal


def accident_type_downstream(damage_level):
    return "LIGHT" if 0 == damage_level else "MEDIUM" if 1 == damage_level else "HEAVY"


class Reporter:
    def __init__(self, appctext, directory, cluster_df, report_interval, initial_date):
        self.appctext = appctext
        self.directory = directory
        self.path = self.appctext.get_resource('adjuster_report_template.docx')
        self.init_date = initial_date
        self.interval = report_interval
        self.mean_weight = 0.8
        self.std_weight = 0.2
        self.topsis_analyzer = TOPSIS(self.appctext, cluster_df=cluster_df, overall=True)
        self.percentile_threshold = 0.5
        self.adjuster_df_reporter = None
        self.criteria = ['File Closure Cost / Vehicle Value',
                         'File Closure Time',
                         'Percentage of Disclaim',
                         'Percentage of Declines']

    def plot_radar_chart(self, adjuster):
        data_adjuster_mean = self.topsis_analyzer.adjuster_df.loc[adjuster, :]
        data_mean = self.topsis_analyzer.adjuster_df.mean()

        radar_chart = pygal.Radar(fill=True)
        radar_chart.title = adjuster + ' Radar Chart'
        radar_chart.x_labels = list(self.topsis_analyzer.adjuster_df.columns)

        radar_chart.add(adjuster, list(np.round(100 * data_adjuster_mean.values, 2)))
        radar_chart.add('Mean', list(np.round(100 * data_mean.values, 2)))

        radar_chart.render_to_file(self.directory + '\\' + adjuster + '.svg')

    def overall_scoring_results(self):
        self.results_df, eliminated_adjusters = self.topsis_analyzer.topsis_analysis()

    def create_adjuster_personal_info_block(self, adjuster, report_context):
        report_context["adjuster_name"] = adjuster
        report_context["adjuster_cities"] = ", ".join(
            set(self.topsis_analyzer.topsis_df[self.topsis_analyzer.topsis_df["ADJUSTER"] == adjuster][
                    "SERVICE CITY"]))
        report_context["interval"] = self.interval

    def create_adjuster_general_info_block(self, adjuster, report_context):
        report_context["adjuster_position"] = self.results_df.at[adjuster, "Ranking"]
        report_context["adjuster_percentile"] = '% ' + str(int(stats.percentileofscore(self.results_df["Ranking"],
                                                                                     self.results_df.at[
                                                                                         adjuster, "Ranking"])
                                                             ))
        reject_rate = '% ' + str(int(100 *
                                     self.topsis_analyzer.topsis_df[self.topsis_analyzer.topsis_df["ADJUSTER"] == adjuster][
                                         "PERCENTAGE OF DECLINES"].unique()[0]))
        report_context["adjuster_reject_rate"] = reject_rate
        self.plot_radar_chart(adjuster)

    def create_adjuster_experience_block(self, adjuster, report_context):
        report_context["number_of_clusters"] = len(self.topsis_analyzer.cluster_df["ACCIDENT TYPE"].unique())
        report_context["accident_types"] = []
        accident_types = sorted(
            self.topsis_analyzer.topsis_df[self.topsis_analyzer.topsis_df["ADJUSTER"] == adjuster]["ACCIDENT TYPE"].unique())

        for accident_type in accident_types:
            accident_type_temp_df = self.topsis_analyzer.topsis_df[
                (self.topsis_analyzer.topsis_df["ADJUSTER"] == adjuster) & (
                        self.topsis_analyzer.topsis_df["ACCIDENT TYPE"] == accident_type)]

            temp_dict = {"type": str(int(accident_type)), "count": len(accident_type_temp_df),
                         "brand": ", ".join(set(accident_type_temp_df["BRAND_NAME"])),
                         "vtype": ", ".join(set(accident_type_temp_df["VEHICLE_USAGE_TYPE"])),
                         "damage": ", ".join(set([accident_type_downstream(element)
                                                 for element in self.topsis_analyzer.cluster_df[
                                                     self.topsis_analyzer.cluster_df["FILE ID"].isin(
                                                         accident_type_temp_df["FILE ID"])][
                                                     "ESTIMATED DAMAGE COST"]]))}

            report_context["accident_types"].append(temp_dict)

    def create_adjuster_system_insight_block(self, adjuster, report_context):

        self.adjuster_df_reporter = pd.DataFrame(np.zeros(self.topsis_analyzer.adjuster_df.shape))
        self.adjuster_df_reporter.index = self.topsis_analyzer.adjuster_df.index
        self.adjuster_df_reporter.columns = self.topsis_analyzer.adjuster_df.columns

        for i, criterion in enumerate(self.criteria):
            mean_criterion = self.topsis_analyzer.adjuster_df[
                self.topsis_analyzer.criterion_converter[criterion]].mean()
            temp = self.topsis_analyzer.adjuster_df[self.topsis_analyzer.criterion_converter[criterion]].values
            self.adjuster_df_reporter[criterion] = np.where(temp < mean_criterion, "altında", "üstünde")

            position = self.adjuster_df_reporter.at[adjuster, criterion]
            if position == "below":
                report_context["adjuster_insight" + str(
                    i + 1)] = adjuster + ", " + criterion + " for the overall achievement is below average. Adjuster can improve herself on this criterion."
            else:
                report_context["adjuster_insight" + str(
                    i + 1)] = adjuster + ", " + criterion + " is above average for the overall achievement."

    def create_report(self):
        # Get overall scores
        self.overall_scoring_results()
        temp_excel = self.results_df.reset_index()
        temp_excel.drop("Sıralama", axis=1, inplace=True)
        temp_excel = temp_excel[["ADJUSTER", "Mean Score", "Confidence Interval", "City"]]
        temp_excel.to_excel(self.directory + '\\adjuster_' + self.interval + '.xlsx', index=False, header=True)
        temp_excel.to_csv(os.getcwd() + '\\adjuster_' + self.interval + '.csv', index=False, header=True)
        clear_console = lambda: os.system('cls' if os.name == 'nt' else 'clear')

        print("Reports are being printed...")

        for adjuster in self.results_df.index:
            print(adjuster + " 's report is being printed...")

            doc = DocxTemplate(self.path)
            report_context = {"date1": self.init_date.strftime("%d/%m/%Y"),
                              "date2": datetime.datetime.now().strftime("%d/%m/%Y")}

            self.create_adjuster_personal_info_block(adjuster, report_context)
            self.create_adjuster_general_info_block(adjuster, report_context)
            self.create_adjuster_experience_block(adjuster, report_context)
            self.create_adjuster_system_insight_block(adjuster, report_context)

            doc.render(report_context, jinja2.Environment(autoescape=True))
            doc.save(self.directory + '\\' + adjuster + '.docx')

            convert(self.directory + '\\' + adjuster + '.docx',
                    self.directory + '\\' + adjuster + '.pdf')
            clear_console()
        return self.results_df
