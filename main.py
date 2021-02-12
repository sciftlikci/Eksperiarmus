from fbs_runtime.application_context.PyQt5 import ApplicationContext
import PyQt5
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from dateutil.relativedelta import relativedelta
from Reporter import *
from ML import *
from TOPSIS import *
from TOPSIS_UI import *
from Transformer import *
from pylab import rcParams
import sys
import cv2
import numpy as np
import warnings
import pkg_resources.py2_warn
from pandas.core.common import flatten
from os import path
from Errors import PermissionDeniedError
import os

global appctext


class SecondWindow(QDialog):
    def __init__(self, parent, *args, **kwargs):
        super(SecondWindow, self).__init__(parent, *args, **kwargs)
        self.selection_parameter = 0
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowTitle("Select Report Time Period")

        self.labeled_slider = LabeledSlider(0, 3, 1,
                                            labels=["Quarterly", "Semi-Annually", "Annually", "Overall"],
                                            orientation=Qt.Horizontal)

        layout = QHBoxLayout()
        layout.addWidget(self.labeled_slider)
        buttonBox = QtWidgets.QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.selected)
        buttonBox.rejected.connect(self.rejected)
        layout.addWidget(buttonBox)
        self.setLayout(layout)
        self.show()

    def selected(self):
        self.close()
        self.selection_parameter += 1
        return self.labeled_slider.sl.value()

    def rejected(self):
        self.close()
        return self.labeled_slider.sl.value()


class Ui_PyTOP(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.pbar = None
        self.pytops_params_df = pd.DataFrame(
            columns=["attributes", "data_pt_threshold", "selected_criteria",
                     "selected_cluster"])

        self.clear_console = lambda: os.system('cls' if os.name == 'nt' else 'clear')

        try:
            self.default_df = pd.read_csv(appctext.get_resource('adjuster_overall_ranking.csv'))
        except PermissionError:
            raise PermissionDeniedError("Related files cannot be opened. Please run the program as an administrator.")

        with open(appctext.get_resource('last_clustering_date.pickle'), 'rb') as handle:
            self.last_clustering_date = datetime.datetime.strptime(pickle.load(handle), "%d/%m/%Y")

        self.cboxes = []
        self.slider_dict = {'File Closure Cost / Vehicle Value': 0,
                            'File Closure Time': 0,
                            'Percentage of Disclaim': 0,
                            'Percentage of Declines': 0}

        self.topsis_df = pd.read_csv(appctext.get_resource('topsis.csv'))
        self.cluster_df = pd.read_csv(appctext.get_resource('clustering.csv'))
        self.city_df = pd.read_csv(appctext.get_resource('city_info.csv'))
        self.city_distance_df = pd.read_csv(appctext.get_resource('city_distance_info.csv'))
        self.price_df = pd.read_csv(appctext.get_resource('vehicle_values.csv'))

        try:
            self.new_accidents_df = pd.read_excel(appctext.get_resource('pre_accident.xlsx'))
        except:
            self.new_accidents_df = None

        # helper objects
        self.transformer = DataTransformer(appctext)
        self.transformer.city_df = self.city_df
        self.transformer.price_df = self.price_df

        cls = load(appctext.get_resource('best_classifier_f1_macro.joblib'))
        self.classifier = ExtraTrees(appctext, self.cluster_df, self.transformer, cls)
        self.clusterer = KMedoidsClusterer(appctext, self.cluster_df, self.transformer)

        self.best_k = 0
        self.eliminated_adjusters = []
        self.msg_date_filter = ''
        self.msg_date_filter_list = []
        self.selected_cluster = None
        self.reset_table = False

        self.labeled_slider_dict = {0: "Quarterly",
                                    1: "Semi-Annually",
                                    2: "Annually",
                                    3: "Overall"
                                    }

        try:
            # read file
            self.new_topsis_df = pd.read_excel(appctext.get_resource('kaza_sonrası.xlsx'))

            # outlier elimination
            self.new_topsis_df = eliminate_outliers(self.new_topsis_df)
            self.new_topsis_df["FILE / INSURANCE_TYPE_2"] = self.transformer.insurance_value_transform_outstream(
                self.new_topsis_df)
            self.new_topsis_df["FILE CLOSURE TIME"] = datetime_transform(self.new_topsis_df)
            self.new_topsis_df = self.new_topsis_df[self.topsis_df.columns]
            self.topsis_df = pd.concat([self.topsis_df, self.new_topsis_df], axis=0, ignore_index=True)
            self.topsis_df.drop_duplicates("FILE ID", inplace=True)

            self.topsis_df.to_csv(appctext.get_resource('topsis.csv'), index=False, header=True)
        except:
            pass

    def month_check(self):
        current_date = datetime.datetime.strptime(datetime.datetime.now().strftime("%d/%m/%Y"), "%d/%m/%Y")
        return (current_date.year - self.last_clustering_date.year) * 12 + (
                current_date.month - self.last_clustering_date.month)

    def pbar_update(self, kind):
        kind_dict = {
            "extra_big": 40,
            "big": 20,
            "medium": 10,
            "small": 5,
            "extra_small": 1
        }
        self.pbar.update(kind_dict[kind])

    def get_info_message_text_by_code(self, message_code):

        msg = QMessageBox()
        msg.setWindowTitle("Adjuster Evaluation System")

        switcher = {
            1: "The result is saved.\nNumber of optimal accident types: " + str(self.best_k),
            2: "Type of accident to be examined: " + str(self.selected_cluster),
            3: "Classification was made and the result was recorded.\nType of accident to be examined: " + str(self.selected_cluster),
            4: ', '.join(self.eliminated_adjusters) + " are excluded due to insufficient number of files." if len(
                self.eliminated_adjusters) > 0 else "All adjusters were included in the ranking procedure.",
            5: "The table has been saved.",
            6: "The operation has been canceled.",
            7: "The report is being prepared. Please wait...",
            8: "Reports are being prepared. Please wait...",
            9: "At least one criterion must be selected to rank the adjusters.",
            10: "First you have to classify an accident file.",
            11: "Please enter a file id.",
            12: "Adjuster overall ranking table was prepared and reports were printed.",
            13: "Error! Operation failed.",
            14: "pre_accident.xlsx cannot be found. Please check the folder and make sure the file is placed.",
            16: self.msg_date_filter + ' Adjuster reports cannot be created due to insufficient data.\n\nAvailable '
                                       'Options: ' + ', '.join(
                set(self.msg_date_filter_list)),
            17: self.msg_date_filter + ' reports were not found. Please first ' + self.msg_date_filter + ' generate '
                                                                                                     'reports. '
        }
        msg.setText(switcher.get(message_code, "Error! Operation failed."))
        if message_code in [9, 10, 11, 17]:
            msg.setIcon(QMessageBox.Warning)
        elif message_code in [16]:
            msg.setIcon(QMessageBox.Critical)
        else:
            msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def setupUi(self, PyTOP):
        PyTOP = palette_creator(PyTOP)

        # Input side
        # QGridLayout
        self.gridLayout = QtWidgets.QGridLayout(PyTOP)
        self.gridLayout.setObjectName("gridLayout")

        # Top Label Block
        PyTOP, self.gridLayout, self.verticalLayout_1, self.verticalLayout_2, self.label_1, self.label_2 = top_label_block(
            PyTOP, self.gridLayout)

        # Clustering block
        PyTOP, self.verticalLayout_2, self.Push_Btn_RUN_CLUSTERING = clustering_block(PyTOP,
                                                                                      self.verticalLayout_2)

        # New Accidents Block
        PyTOP, self.verticalLayout_2, self.label_3_1, self.label_4, self.Text_accident_params = new_accidents_block(
            PyTOP,
            self.verticalLayout_2)

        # Run Classification
        PyTOP, self.verticalLayout_2, self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT = run_classification_button(PyTOP,
                                                                                                                self.verticalLayout_2)

        # Data point threshold block start
        PyTOP, self.verticalLayout_2, self.label_6, self.spinBox_threshold = data_point_threshold_block(PyTOP,
                                                                                                        self.verticalLayout_2)

        # Criteria Selection Block start
        PyTOP, self.gridLayout, self.label_7, self.cboxes, self.Push_Btn_RUN = criteria_selection_block(PyTOP,
                                                                                                        self.verticalLayout_2,
                                                                                                        self.gridLayout)
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)

        # Output side
        # Output label block
        PyTOP, self.verticalLayout_3, self.label_8 = output_label_block(PyTOP)

        # Run button
        # PyTOP, self.verticalLayout_3, self.Push_Btn_RUN = run_button_block(PyTOP, self.verticalLayout_3)

        # adjuster ranking block
        PyTOP, self.verticalLayout_3, self.table_mean_relative_closeness, self.horizontalHeader, self.model, self.proxy = table_mean_relative_closeness_block(
            PyTOP, self.verticalLayout_3, self.default_df)

        # adjuster scores save block
        PyTOP, self.verticalLayout_3, self.Tool_Btn_save_mean_relative_closeness, self.Push_Btn_RESET, self.Push_Btn_TIME_REPORT_TABLE = save_table_mean_relative_closeness_block(
            PyTOP, self.verticalLayout_3)

        ## adjuster Report Combo
        PyTOP, self.verticalLayout_3, self.label_12 = print_adjuster_report_combo_block(
            PyTOP, self.verticalLayout_3)

        # adjuster Report Plot Button
        PyTOP, self.verticalLayout_3, self.gridLayout, self.Push_Btn_print_adjuster_report = print_adjuster_report_button_block(
            PyTOP, self.verticalLayout_3, self.gridLayout)

        PyTOP, self.gridLayout = draw_line(PyTOP, self.gridLayout)

        self.retranslateUi(PyTOP)
        QtCore.QMetaObject.connectSlotsByName(PyTOP)

        self.PyTOP = PyTOP

    def retranslateUi(self, PyTOP):
        _translate = QtCore.QCoreApplication.translate
        PyTOP.setWindowTitle(_translate("Adjuster Evaluation System", "Adjuster Evaluation System"))

        # Top Label
        self.label_1.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; "
                                        "font-weight:600;\">Evaluation System</span></p><p align=\"center\"><span style=\" "
                                        "font-size:14pt; font-style:italic;\">Adjuster Performance Evaluation and Management System</span></p></body></html>"))
        # Inputs
        self.label_2.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                        "font-weight:600;\">Create Accident Type (Last Created Date: " + self.last_clustering_date.strftime(
                                            "%d/%m/%Y") + ")</span></p></body></html>"))

        self.Push_Btn_RUN_CLUSTERING.setText(_translate("PyTOP", "Start Creating Accident Type"))

        self.label_3_1.setText(_translate("PyTOP",
                                          "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                          "font-weight:600;\">Determining the Type of Accident</span></p></body></html>"))

        # New Damage Files
        self.label_4.setText(_translate("PyTOP",
                                        "<html><head/><body><p><h4>File ID</h4></p></body></html>"))
        self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setText(_translate("PyTOP", "Determine Accident Type"))
        # Decision Matrix
        self.label_6.setText(_translate("PyTOP",
                                        "<html><head/><body><p><h4>File Count Threshold</h4></p></body></html>"))
        # Criteria checkbox
        self.label_7.setText(_translate("PyTOP",
                                        "<html><head/><body><p><h4>Criteria Selection</h4></p></body></html>"))

        self.label_8.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                        "font-weight:600;\">Adjuster Overall Rankings</span></p></body></html>"))

        self.Push_Btn_RUN.setText(_translate("PyTOP", "Rank Adjusters"))
        self.Tool_Btn_save_mean_relative_closeness.setText(_translate("PyTOP", "Save Table"))
        self.Push_Btn_RESET.setText(_translate("PyTOP", "Return to Overall Rankings"))
        self.Push_Btn_TIME_REPORT_TABLE.setText(_translate("PyTOP", "Select Ranking"))
        self.label_12.setText(_translate("PyTOP",
                                         "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                         "font-weight:600;\">Adjuster Report Cards</span></p></body></html>"))

        self.Push_Btn_print_adjuster_report.setText(_translate("PyTOP", "Print Adjuster Report Cards"))

        # File Choosing buttons
        self.Push_Btn_RUN_CLUSTERING.clicked.connect(self.run_clustering)
        self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.clicked.connect(self.run_classification_for_new_accident)
        self.Push_Btn_RUN.clicked.connect(self.run_pytops)
        self.Push_Btn_RESET.clicked.connect(self.reset_default_table)
        self.Push_Btn_TIME_REPORT_TABLE.clicked.connect(self.show_time_report_table)
        self.Push_Btn_print_adjuster_report.clicked.connect(self.print_report)
        self.Text_accident_params.textChanged.connect(self.initial_green_button)
        self.horizontalHeader.sectionClicked.connect(self.on_view_horizontalHeader_sectionClicked)

        # File Saving Buttons
        self.Tool_Btn_save_mean_relative_closeness.clicked.connect(self.save_mean)

    @QtCore.pyqtSlot(int)
    def on_view_horizontalHeader_sectionClicked(self, logicalIndex):

        self.logicalIndex = logicalIndex
        self.menuValues = QtWidgets.QMenu(self)
        self.signalMapper = QtCore.QSignalMapper(self)
        self.city_filtering = False

        valuesUnique = [self.model.item(row, self.logicalIndex).text()
                        for row in range(self.model.rowCount())
                        ]

        if any(item in list(self.city_df["City"]) for item in valuesUnique):
            self.city_filtering = True
            letters = "abcçdefgğhıi̇jklmnoöprsştuüvyz"

            valuesUnique = [elem.split(",") for elem in valuesUnique]
            valuesUnique = list(flatten(valuesUnique))
            valuesUnique = [elem.strip() for elem in valuesUnique if elem != '']
            valuesUnique = sorted(set(valuesUnique), key=lambda i: (letters.index(i[0].lower()),
                                                                    letters.index(i[1].lower()),
                                                                    letters.index(i[2].lower())))

        actionAll = QtWidgets.QAction("All", self)
        actionAll.triggered.connect(self.on_actionAll_triggered)
        self.menuValues.addAction(actionAll)
        self.menuValues.addSeparator()
        for actionNumber, actionName in enumerate(sorted(list(set(valuesUnique)))):
            action = QtWidgets.QAction(actionName, self)
            self.signalMapper.setMapping(action, actionNumber)
            action.triggered.connect(self.signalMapper.map)
            self.menuValues.addAction(action)
        self.signalMapper.mapped.connect(self.on_signalMapper_mapped)
        headerPos = self.table_mean_relative_closeness.mapToGlobal(self.horizontalHeader.pos())
        posY = headerPos.y() + self.horizontalHeader.height()
        posX = headerPos.x() + self.horizontalHeader.sectionPosition(self.logicalIndex)

        self.menuValues.exec_(QtCore.QPoint(posX, posY))

    @QtCore.pyqtSlot()
    def on_actionAll_triggered(self):
        filterColumn = self.logicalIndex
        filterString = QtCore.QRegExp("",
                                      QtCore.Qt.CaseInsensitive,
                                      QtCore.QRegExp.RegExp
                                      )

        self.proxy.setFilterRegExp(filterString)
        self.proxy.setFilterKeyColumn(filterColumn)

    @QtCore.pyqtSlot(int)
    def on_signalMapper_mapped(self, i):
        stringAction = self.signalMapper.mapping(i).text()
        filterColumn = self.logicalIndex
        if not self.city_filtering:
            filterString = QtCore.QRegExp(r'^' + stringAction + '$',
                                          QtCore.Qt.CaseSensitive,
                                          QtCore.QRegExp.RegExp2
                                          )
        else:
            filterString = QtCore.QRegExp(stringAction,
                                          QtCore.Qt.CaseSensitive,
                                          QtCore.QRegExp.RegExp2
                                          )
        self.proxy.setFilterRegExp(filterString)
        self.proxy.setFilterKeyColumn(filterColumn)

    @QtCore.pyqtSlot(int)
    def on_comboBox_currentIndexChanged(self, index):
        self.proxy.setFilterKeyColumn(index)

    def initial_green_button(self):
        try:
            if str(self.Text_accident_params.displayText()) != '' and int(
                    self.Text_accident_params.displayText()) not in self.new_accidents_df["FILE ID"].values and int(
                self.Text_accident_params.displayText()) not in self.cluster_df["FILE ID"].values:
                self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: red")
            elif str(self.Text_accident_params.displayText()) == '':
                self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: lightgray")
            else:
                self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: green")
        except:
            pass

    def show_time_report_table(self):

        len_checked = True

        while len_checked:
            dialog = SecondWindow(self)
            dialog.exec_()
            if dialog.selection_parameter > 0:
                self.msg_date_filter = self.labeled_slider_dict[dialog.labeled_slider.sl.value()]
                try:
                    self.time_report_table_df = pd.read_csv(
                        os.getcwd() + '\\adjuster_' + self.msg_date_filter + '_ranking.csv')
                    len_checked = False
                    _translate = QtCore.QCoreApplication.translate
                    self.label_8.setText(_translate("PyTOP",
                                                    "<html><head/><body><p align=\"center\"><span style=\" "
                                                    "font-size:14pt; "
                                                    "font-weight:600;\">Adjuster " + self.msg_date_filter + " Rankings</span></p></body></html>"))
                except:
                    for i in range(dialog.labeled_slider.sl.value() + 1, 4):
                        try:
                            self.recommend_filter = pd.read_csv(
                                os.getcwd() + '\\adjuster_' + self.msg_date_filter + '_rankings.csv')
                            if len(self.topsis_copy_df) != 0:
                                self.msg_date_filter_list.append(self.labeled_slider_dict[i])
                        except:
                            pass
                    self.get_info_message_text_by_code(17)
            else:
                self.get_info_message_text_by_code(6)
                return

        self.model.clear()
        self.model.setRowCount(len(self.time_report_table_df))
        self.model.setColumnCount(len(self.time_report_table_df.columns))

        for n, col in enumerate(list(self.time_report_table_df.columns)):
            for m, item in enumerate(self.time_report_table_df[col]):
                item = QtGui.QStandardItem(str(item))
                item.setTextAlignment(QtCore.Qt.AlignHCenter)
                self.model.setItem(m, n, item)

        self.model.setHorizontalHeaderLabels(list(self.time_report_table_df.columns))
        self.table_mean_relative_closeness.resizeColumnsToContents()
        self.table_mean_relative_closeness.resizeRowsToContents()
        self.proxy.setSourceModel(self.model)
        self.table_mean_relative_closeness.setModel(self.proxy)

    def reset_default_table(self):
        self.display_results(reset_table=True)
        self.selected_cluster = None
        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(_translate("PyTOP",
                                        "<html><head/><body><p><h4>File ID</h4></p></body></html>"))
        self.label_8.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                        "font-weight:600;\">Adjuster Overall Rankings</span></p></body></html>"))
        self.Push_Btn_RUN.setStyleSheet("background-color: lightgray")
        self.Text_accident_params.setText("")

    def cluster_and_classify_data(self):
        messagebox = TimerMessageBox(1, 2, self)
        messagebox.exec_()

        try:
            self.best_k = self.clusterer.fit(self.cluster_df)
        except Exception as e:
            self.get_info_message_text_by_code(13)
            return

        messagebox = TimerMessageBox(2, 2, self)
        messagebox.exec_()

        self.classifier.fit(self.clusterer.clustering_df[self.clusterer.clustering_df.columns[:-2]],
                            self.clusterer.clustering_df["ACCIDENT TYPE"])
        self.get_info_message_text_by_code(1)

    def run_clustering(self):

        self.clear_console()

        # Clustering part
        month_difference = self.month_check()

        if month_difference < 6:
            # Get final decision from user based on last clustering date

            buttonReply = QMessageBox()
            buttonReply.setIcon(QMessageBox.Question)
            buttonReply.setWindowTitle('Evaluation System')
            buttonReply.setText("It has been " + str(
                month_difference) + " month since the last accident type was created.\nWould you like to create them anyway? (Not recommended before 6 months.)")
            buttonReply.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            buttonY = buttonReply.button(QMessageBox.Yes)
            buttonY.setText('Yes')
            buttonN = buttonReply.button(QMessageBox.No)
            buttonN.setText('Cancel')
            buttonReply.exec_()

            if buttonReply.clickedButton() == buttonY:
                self.cluster_and_classify_data()
            else:
                self.get_info_message_text_by_code(6)
        else:
            self.cluster_and_classify_data()

    def run_classification_for_new_accident(self):

        self.clear_console()

        if self.new_accidents_df is None:
            self.get_info_message_text_by_code(14)
            return

        if len(self.Text_accident_params.displayText()) == 0:
            self.get_info_message_text_by_code(11)
            return

        if int(self.Text_accident_params.displayText()) not in self.cluster_df["FILE ID"].values:

            # message box for starting
            messagebox = TimerMessageBox(3, 2, self)
            messagebox.exec_()

            self.accident_params = self.new_accidents_df[
                self.new_accidents_df["FILE ID"] == int(self.Text_accident_params.displayText())]
            self.accident_params = self.accident_params[
                ["INSURANCE_TYPE", "SERVICE_TYPE", "ESTIMATED DAMAGE COST", "SERVICE CITY", "BRAND_NAME",
                 "MODEL_NAME"]].values[0].tolist()

            # QThread for progress bar
            self.pbar = tqdm(total=100, position=0, leave=True, desc="The new accident is being classified ...")
            self.accident_params[2] = float(self.accident_params[2])
            self.pbar_update("extra_big")
            self.accident_city = self.accident_params[3]
            self.pbar_update("extra_big")

            # get cluster label for new accident
            self.selected_cluster = self.classifier.predict(self.accident_params,
                                                            int(self.Text_accident_params.displayText()))
            self.pbar_update("big")
            if self.selected_cluster is None:
                return

            # save new accident
            self.get_info_message_text_by_code(3)
        else:
            temp_accident_params_df = self.cluster_df[
                self.cluster_df["FILE ID"] == int(self.Text_accident_params.displayText())]
            self.accident_city = \
                self.city_df[self.city_df["SosyoEko"] == float(temp_accident_params_df["SERVICE CITY"].values[0])][
                    "City"].values[0]
            self.selected_cluster = int(temp_accident_params_df["ACCIDENT TYPE"].values[0])
            self.get_info_message_text_by_code(2)

        _translate = QtCore.QCoreApplication.translate

        self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: lightgray")
        self.Push_Btn_RUN.setStyleSheet("background-color: green")

    def display_results(self, overall=False, reset_table=False):
        if not reset_table:
            self.results_df.reset_index(inplace=True)
            self.table_df = self.results_df[
                ["ADJUSTER", "Mean Score", "Confidence Interval", "City", "Available"]] if not overall else \
                self.results_df[["ADJUSTER", "Mean Score", "Confidence Interval", "City"]]
        else:
            self.table_df = self.default_df
            self.reset_table = True
            self.Push_Btn_RESET.setStyleSheet("background-color: lightgray")

        self.model.clear()
        self.model.setRowCount(len(self.table_df))
        self.model.setColumnCount(len(self.table_df.columns))

        for n, col in enumerate(list(self.table_df.columns)):
            for m, item in enumerate(self.table_df[col]):
                item = QtGui.QStandardItem(str(item))
                item.setTextAlignment(QtCore.Qt.AlignHCenter)
                self.model.setItem(m, n, item)

        self.model.setHorizontalHeaderLabels(list(self.table_df.columns))
        self.table_mean_relative_closeness.resizeColumnsToContents()
        self.table_mean_relative_closeness.resizeRowsToContents()
        self.proxy.setSourceModel(self.model)
        self.table_mean_relative_closeness.setModel(self.proxy)

    def report_date_filter(self, report_interval):
        now = datetime.datetime.now()

        date_switcher = {
            0: now - relativedelta(months=3),
            1: now - relativedelta(months=6),
            2: now - relativedelta(months=12),
            3: now - relativedelta(years=100)
        }
        self.topsis_copy_df = self.topsis_df.copy()
        self.topsis_copy_df["ADJUSTER_ASSIGN_DATE"] = pd.to_datetime(self.topsis_copy_df["ADJUSTER_ASSIGN_DATE"])
        self.topsis_copy_df["REPORT_DATE"] = pd.to_datetime(self.topsis_copy_df["REPORT_DATE"])

        self.initial_date = date_switcher[report_interval]
        self.topsis_copy_df = self.topsis_copy_df[
            (self.topsis_copy_df["ADJUSTER_ASSIGN_DATE"] >= self.initial_date) & (
                    self.topsis_copy_df["REPORT_DATE"] >= self.initial_date)]

        return self.labeled_slider_dict[report_interval]

    def print_report(self):

        self.clear_console()

        len_checked = True

        while len_checked:
            dialog = SecondWindow(self)
            dialog.exec_()
            if dialog.selection_parameter > 0:
                self.msg_date_filter = self.report_date_filter(dialog.labeled_slider.sl.value())
                if len(self.topsis_copy_df) != 0:
                    len_checked = False
                    self.topsis_df = self.topsis_copy_df
                else:
                    for i in range(dialog.labeled_slider.sl.value() + 1, 4):
                        self.recommend_filter = self.report_date_filter(i)
                        if len(self.topsis_copy_df) != 0:
                            self.msg_date_filter_list.append(self.recommend_filter)
                    self.get_info_message_text_by_code(16)
            else:
                self.get_info_message_text_by_code(6)
                return

        directory = str(QFileDialog.getExistingDirectory(self, "Save Reports"))

        if directory == '':
            self.get_info_message_text_by_code(6)
            return

        messagebox = TimerMessageBox(5, 3, self)
        messagebox.exec_()

        reporter = Reporter(appctext, directory, self.cluster_df, self.msg_date_filter, self.initial_date)

        reporter.topsis_analyzer.transformer = self.transformer
        reporter.topsis_analyzer.topsis_df = self.topsis_df
        reporter.topsis_analyzer.city_distance_df = self.city_distance_df
        reporter.topsis_analyzer.city_df = self.city_df

        self.results_df = reporter.create_report()
        self.get_info_message_text_by_code(12)

        self.display_results(overall=True)
        _translate = QtCore.QCoreApplication.translate
        self.label_8.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                        "font-weight:600;\">Adjuster Overall Rankings</span></p></body></html>"))
        self.Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: lightgray")
        self.Push_Btn_RUN.setStyleSheet("background-color: lightgray")
        self.Push_Btn_RESET.setStyleSheet("background-color: lightgray")

    # File Saving Functions
    def save_mean(self):
        dialog_path = QFileDialog.getSaveFileName(self, 'Save Table', '', '*.xlsx')

        if dialog_path[0] == '':
            self.get_info_message_text_by_code(6)
            return

        elif "Genel" not in self.label_8.text():
            model = self.table_mean_relative_closeness.model()
            save_table_df = pd.DataFrame(0, index=np.arange(model.rowCount()),
                                         columns=[
                                             self.table_mean_relative_closeness.model().headerData(i, Qt.Horizontal) for
                                             i in range(model.columnCount())])

            for row in range(model.rowCount()):
                for column in range(model.columnCount()):
                    index = model.index(row, column)
                    # We suppose data are strings
                    save_table_df.iloc[row, column] = model.data(index)

            save_table_df.to_excel(dialog_path[0], index=False, header=True)
            self.get_info_message_text_by_code(5)
        else:
            self.default_df.to_excel(dialog_path[0], index=False, header=True)
            self.get_info_message_text_by_code(5)

    def transform_user_inputs(self):
        self.pytops_params_df["selected_criteria"] = [[cbox.text() for cbox in self.cboxes if cbox.isChecked()]]
        self.pytops_params_df["attributes"] = [
            [self.slider_dict[cbox.text()] for cbox in self.cboxes if cbox.isChecked()]]
        self.pytops_params_df["selected_cluster"] = self.selected_cluster
        self.pytops_params_df["data_pt_threshold"] = int(self.spinBox_threshold.cleanText())

    def run_pytops(self):

        self.clear_console()

        """
        Run TOPSIS study with user-defined parameters.
        """

        if self.selected_cluster is None:
            self.get_info_message_text_by_code(10)
            return

        if len([cbox for cbox in self.cboxes if cbox.isChecked()]) == 0:
            self.get_info_message_text_by_code(9)
            return

        # message box for starting analysis
        self.messagebox = TimerMessageBox(4, 2, self)
        self.messagebox.exec_()
        self.reset_table = False

        # Get user inputs from buttons and dropdown lists
        self.transform_user_inputs()

        # Run TOPSIS analysis
        self.topsis_analyzer = TOPSIS(appctext, self.cluster_df, self.pytops_params_df, self.selected_cluster)
        self.topsis_analyzer.file_id = int(self.Text_accident_params.displayText())
        self.topsis_analyzer.topsis_df = self.topsis_df
        self.topsis_analyzer.transformer = self.transformer
        self.topsis_analyzer.city_distance_df = self.city_distance_df

        self.results_df, self.eliminated_adjusters = self.topsis_analyzer.topsis_analysis(self.accident_city)

        if self.results_df is None or self.eliminated_adjusters is None:
            return

        # Warn user for eliminated adjusters from current study
        self.get_info_message_text_by_code(4)

        # Prepare results as text to display
        self.display_results()
        # Plotting and saving results' functions will be triggered after pressing their buttons.

        _translate = QtCore.QCoreApplication.translate
        self.label_8.setText(_translate("PyTOP",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; "
                                        "font-weight:600;\">Adjuster Accident Type Ranking (Chosen Accident Type No: " + str(
                                            self.selected_cluster) + ")</span></p></body></html>"))

        self.Push_Btn_RUN.setStyleSheet("background-color: lightgray")
        self.Push_Btn_RESET.setStyleSheet("background-color: lightblue")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    clear_console = lambda: os.system('cls' if os.name == 'nt' else 'clear')
    clear_console()
    print("Adjuster Evaluation System is running. Please wait...")
    appctext = ApplicationContext()
    Form = QtWidgets.QWidget()
    ui = Ui_PyTOP()
    ui.setupUi(Form)
    Form.showMaximized()
    sys.exit(appctext.app.exec_())
