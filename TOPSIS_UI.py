from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyautogui
import os
import time
import pandas as pd
from pandas.core.common import flatten
import numpy as np
from PyQt5.QtGui import *


class QHSeperationLine(QtWidgets.QFrame):
    """
    a horizontal seperation line\n
    """
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        # self.setFrameShadow(QtWidgets.QFrame.Sunken)
        return


class LabeledSlider(QWidget):
    def __init__(self, minimum, maximum, interval=1, orientation=Qt.Horizontal,
                 labels=None, p0=0, parent=None):
        super(LabeledSlider, self).__init__(parent=parent)

        levels = range(minimum, maximum + interval, interval)

        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels = list(zip(levels, labels))
        else:
            self.levels = list(zip(levels, map(str, levels)))

        if orientation == Qt.Horizontal:
            self.layout = QVBoxLayout(self)
        elif orientation == Qt.Vertical:
            self.layout = QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin = 10
        self.top_margin = 20
        self.right_margin = 10
        self.bottom_margin = 20

        self.layout.setContentsMargins(self.left_margin, self.top_margin,
                                       self.right_margin, self.bottom_margin)

        self.sl = QSlider(orientation, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setSliderPosition(p0)
        if orientation == Qt.Horizontal:
            self.sl.setTickPosition(QSlider.TicksBelow)
            self.sl.setMinimumWidth(300)  # just to make it easier to read
        else:
            self.sl.setTickPosition(QSlider.TicksLeft)
            self.sl.setMinimumHeight(300)  # just to make it easier to read
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)

        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super(LabeledSlider, self).paintEvent(e)
        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:

            # get the size of the label
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            if self.sl.orientation() == Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                       self.sl.maximum(), v, available) + length // 2

                # left bound of the text = center - half of text width + L_margin
                left = x_loc - rect.width() // 2 + self.left_margin
                bottom = self.rect().bottom()

                # enlarge margins if clipping
                if v == self.sl.minimum():
                    if left <= 0:
                        self.left_margin = rect.width() // 2 - x_loc
                    if self.bottom_margin <= rect.height():
                        self.bottom_margin = rect.height()

                    self.layout.setContentsMargins(self.left_margin,
                                                   self.top_margin, self.right_margin,
                                                   self.bottom_margin)

                if v == self.sl.maximum() and rect.width() // 2 >= self.right_margin:
                    self.right_margin = rect.width() // 2
                    self.layout.setContentsMargins(self.left_margin,
                                                   self.top_margin, self.right_margin,
                                                   self.bottom_margin)

            else:
                y_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                       self.sl.maximum(), v, available, upsideDown=True)

                bottom = y_loc + length // 2 + rect.height() // 2 + self.top_margin - 3
                # there is a 3 px offset that I can't attribute to any metric

                left = self.left_margin - rect.width()
                if left <= 0:
                    self.left_margin = rect.width() + 2
                    self.layout.setContentsMargins(self.left_margin,
                                                   self.top_margin, self.right_margin,
                                                   self.bottom_margin)

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

        return


def get_timed_message_text_by_code(message_code):
    switcher = {
        1: "Accident type creation started. Please wait...",
        2: "Accident type creation is completed, classification model is being trained. Please wait...",
        3: "The new accident is classified. Please wait...",
        4: "The experts are being ranked. Please wait...",
        5: "Reports are being printed. Please wait..."
    }
    return switcher.get(message_code, "Error! Operation failed.")


# PyQt Utils

class TimerMessageBox(QMessageBox):
    def __init__(self, message_code, timeout=3, parent=None):
        super(TimerMessageBox, self).__init__(parent)
        self.setWindowTitle("Adjuster Evaluation System")
        self.time_to_wait = timeout
        self.message_text = get_timed_message_text_by_code(message_code)
        self.setText(self.message_text)
        self.setStandardButtons(QMessageBox.NoButton)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.changeContent)
        self.timer.start()

    def changeContent(self):
        self.setText(self.message_text)
        self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


def palette_creator(PyTOP):
    PyTOP.setObjectName("PyTOP")

    palette = QtGui.QPalette()
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(QtGui.QColor(252, 175, 62))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
    brush = QtGui.QBrush(QtGui.QColor(233, 235, 231))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
    brush = QtGui.QBrush(QtGui.QColor(141, 143, 138))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
    brush = QtGui.QBrush(QtGui.QColor(211, 215, 207))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
    brush = QtGui.QBrush(QtGui.QColor(233, 235, 231))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(QtGui.QColor(252, 175, 62))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
    brush = QtGui.QBrush(QtGui.QColor(233, 235, 231))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
    brush = QtGui.QBrush(QtGui.QColor(141, 143, 138))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
    brush = QtGui.QBrush(QtGui.QColor(211, 215, 207))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
    brush = QtGui.QBrush(QtGui.QColor(233, 235, 231))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(QtGui.QColor(252, 175, 62))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
    brush = QtGui.QBrush(QtGui.QColor(233, 235, 231))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
    brush = QtGui.QBrush(QtGui.QColor(141, 143, 138))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
    brush = QtGui.QBrush(QtGui.QColor(105, 107, 103))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
    brush = QtGui.QBrush(QtGui.QColor(211, 215, 207))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
    brush = QtGui.QBrush(QtGui.QColor(211, 215, 207))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
    brush = QtGui.QBrush(QtGui.QColor(211, 215, 207))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
    PyTOP.setPalette(palette)

    return PyTOP


def top_label_block(PyTOP, gridLayout):
    # Top Label Block Start
    verticalLayout_1 = QtWidgets.QVBoxLayout()
    verticalLayout_1.setObjectName("verticalLayout_1")
    label_1 = QtWidgets.QLabel(PyTOP)

    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setPointSize(20)
    font.setBold(True)
    font.setItalic(False)
    font.setWeight(10)
    label_1.setFont(font)
    label_1.setFrameShape(QtWidgets.QFrame.WinPanel)
    label_1.setFrameShadow(QtWidgets.QFrame.Plain)
    label_1.setLineWidth(4)
    label_1.setObjectName("label_1")
    verticalLayout_1.addWidget(label_1)
    gridLayout.addLayout(verticalLayout_1, 0, 0, 1, 3)

    verticalLayout_2 = QtWidgets.QVBoxLayout()
    verticalLayout_2.setObjectName("verticalLayout_2")
    label_2 = QtWidgets.QLabel(PyTOP)
    label_2.setFrameShape(QtWidgets.QFrame.Panel)
    label_2.setObjectName("label_2")
    verticalLayout_2.addWidget(label_2)

    return PyTOP, gridLayout, verticalLayout_1, verticalLayout_2, label_1, label_2


def clustering_block(PyTOP, verticalLayout_2):
    Push_Btn_RUN_CLUSTERING = QtWidgets.QPushButton(PyTOP)
    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setBold(True)
    font.setItalic(False)
    font.setWeight(50)
    Push_Btn_RUN_CLUSTERING.setFont(font)
    Push_Btn_RUN_CLUSTERING.setObjectName("Push_Btn_RUN_CLUSTERING")
    Push_Btn_RUN_CLUSTERING.setStyleSheet("background-color: lightgray")
    Push_Btn_RUN_CLUSTERING.setToolTip(
        "According to the accident preliminary information, all accident files form groups and accident types "
        "to which they are closest. (THIS PROCESS IS LONG)")
    verticalLayout_2.addWidget(Push_Btn_RUN_CLUSTERING)
    verticalLayout_2.addSpacing(30)

    return PyTOP, verticalLayout_2, Push_Btn_RUN_CLUSTERING


def new_accidents_block(PyTOP, verticalLayout_2):
    verticalLayout_2.addSpacing(40)

    label_3_1 = QtWidgets.QLabel(PyTOP)
    label_3_1.setObjectName("label_3_1")
    label_3_1.setFrameShape(QtWidgets.QFrame.Panel)
    verticalLayout_2.addWidget(label_3_1)

    label_4 = QtWidgets.QLabel(PyTOP)
    label_4.setObjectName("label_4")
    verticalLayout_2.addWidget(label_4)

    # Get new accident params
    Text_accident_params = QtWidgets.QLineEdit(PyTOP)
    Text_accident_params.setObjectName("Text_accident_params")
    Text_accident_params.setValidator(QRegExpValidator(QRegExp("[0-9]+.?[0-9]"), Text_accident_params))
    verticalLayout_2.addWidget(Text_accident_params)

    return PyTOP, verticalLayout_2, label_3_1, label_4, Text_accident_params


def run_classification_button(PyTOP, verticalLayout_2):
    Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT = QtWidgets.QPushButton(PyTOP)
    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setBold(True)
    font.setItalic(False)
    font.setWeight(50)
    Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setFont(font)
    Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setObjectName("Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT")
    Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setStyleSheet("background-color: lightgray")
    Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT.setToolTip("Assigns the new accident file to the closest accident type.")
    verticalLayout_2.addWidget(Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT)

    spacerItem1 = QtWidgets.QSpacerItem(40, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    verticalLayout_2.addItem(spacerItem1)

    return PyTOP, verticalLayout_2, Push_Btn_RUN_CLASSIFICATION_NEW_ACCIDENT


def data_point_threshold_block(PyTOP, verticalLayout_2):
    label_6 = QtWidgets.QLabel(PyTOP)
    label_6.setObjectName("label_6")
    label_6.setToolTip("Minimum number of accident files required to rank an adjuster")
    verticalLayout_2.addWidget(label_6)

    spinBox_threshold = QtWidgets.QSpinBox(PyTOP)
    spinBox_threshold.setMaximum(100)
    spinBox_threshold.setSingleStep(1)
    spinBox_threshold.setObjectName("spinBox_threshold")
    verticalLayout_2.addWidget(spinBox_threshold)

    return PyTOP, verticalLayout_2, label_6, spinBox_threshold


def criteria_selection_block(PyTOP, verticalLayout_2, gridLayout):
    # Checkbox block start
    label_7 = QtWidgets.QLabel(PyTOP)
    label_7.setObjectName("label_7")
    label_7.setToolTip("Selection of the criteria to be used to rank the adjusters")
    verticalLayout_2.addWidget(label_7)

    horizontalLayout = QtWidgets.QHBoxLayout()
    horizontalLayout.setObjectName("horizontalLayout")
    verticalLayout_2.addLayout(horizontalLayout)

    crit = ['File Closure Cost / Vehicle Value', 'File Closure Time', 'Percentage of Disclaim', 'Percentage of Declines']
    cboxes = []
    for col in crit:
        cbox = QCheckBox(col)
        cbox.setChecked(True)
        horizontalLayout.addWidget(cbox)
        cboxes.append(cbox)

    verticalLayout_2.addSpacing(10)

    Push_Btn_RUN = QtWidgets.QPushButton(PyTOP)

    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setWeight(50)
    Push_Btn_RUN.setFont(font)
    Push_Btn_RUN.setObjectName("Push_Btn_RUN")
    Push_Btn_RUN.setStyleSheet("background-color: lightgray")
    Push_Btn_RUN.setToolTip(
        "It makes the rankings of the experts in the accident type determined as a result of the classification and "
        "the selected criteria. ")
    verticalLayout_2.addWidget(Push_Btn_RUN)

    return PyTOP, gridLayout, label_7, cboxes, Push_Btn_RUN


def output_label_block(PyTOP):
    verticalLayout_3 = QtWidgets.QVBoxLayout()
    verticalLayout_3.setObjectName("verticalLayout_3")

    label_8 = QtWidgets.QLabel(PyTOP)
    label_8.setFrameShape(QtWidgets.QFrame.Panel)
    label_8.setObjectName("label_8")
    verticalLayout_3.addWidget(label_8)

    return PyTOP, verticalLayout_3, label_8


def run_button_block(PyTOP, verticalLayout_3):
    Push_Btn_RUN = QtWidgets.QPushButton(PyTOP)

    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setWeight(50)
    Push_Btn_RUN.setFont(font)
    Push_Btn_RUN.setObjectName("Push_Btn_RUN")
    Push_Btn_RUN.setStyleSheet("background-color: lightgray")
    verticalLayout_3.addWidget(Push_Btn_RUN)

    return PyTOP, verticalLayout_3, Push_Btn_RUN


def text_Rank_with_varying_weights_block(PyTOP, verticalLayout_3):
    label_9 = QtWidgets.QLabel(PyTOP)
    label_9.setObjectName("label_9")
    verticalLayout_3.addWidget(label_9)
    text_Rank_with_varying_weights = QtWidgets.QTextBrowser(PyTOP)
    text_Rank_with_varying_weights.setObjectName("text_Rank_with_varying_weights")
    verticalLayout_3.addWidget(text_Rank_with_varying_weights)

    return PyTOP, verticalLayout_3, label_9, text_Rank_with_varying_weights


def table_mean_relative_closeness_block(PyTOP, verticalLayout_3, table_df):
    tableWidget = QTableView()
    model = QtGui.QStandardItemModel()

    model.setRowCount(len(table_df))
    model.setColumnCount(len(table_df.columns))

    for n, col in enumerate(list(table_df.columns)):
        for m, item in enumerate(table_df[col]):
            item = QtGui.QStandardItem(str(item))
            item.setTextAlignment(QtCore.Qt.AlignHCenter)
            model.setItem(m, n, item)

    model.setHorizontalHeaderLabels(list(table_df.columns))
    tableWidget.resizeColumnsToContents()
    tableWidget.resizeRowsToContents()
    verticalLayout_3.addWidget(tableWidget)

    proxy = QtCore.QSortFilterProxyModel()
    proxy.setSourceModel(model)
    tableWidget.setModel(proxy)
    horizontalHeader = tableWidget.horizontalHeader()

    return PyTOP, verticalLayout_3, tableWidget, horizontalHeader, model, proxy


def save_table_mean_relative_closeness_block(PyTOP, verticalLayout_3):
    horizontalLayout = QtWidgets.QHBoxLayout()
    horizontalLayout.setObjectName("horizontalLayout")

    Tool_Btn_save_mean_relative_closeness = QtWidgets.QToolButton(PyTOP)
    Tool_Btn_save_mean_relative_closeness.setObjectName("Tool_Btn_save_mean_relative_closeness")
    Tool_Btn_save_mean_relative_closeness.setToolTip("Saves the table above as an Excel file.")
    horizontalLayout.addWidget(Tool_Btn_save_mean_relative_closeness)

    Push_Btn_RESET = QtWidgets.QPushButton(PyTOP)
    Push_Btn_RESET.setToolTip("Returns to the general adjuster ranking table and resets the selected accident type.")
    Push_Btn_RESET.setObjectName("Push_Btn_RESET")
    Push_Btn_RESET.setStyleSheet("background-color: lightgray")
    Push_Btn_RESET.setMaximumWidth(200)
    horizontalLayout.addWidget(Push_Btn_RESET)

    Push_Btn_REPORT = QtWidgets.QPushButton(PyTOP)
    Push_Btn_REPORT.setToolTip("Shows the reports of the adjusters created according to the selected time interval.")
    Push_Btn_REPORT.setObjectName("Push_Btn_REPORT")
    Push_Btn_REPORT.setStyleSheet("background-color: lightgray")
    Push_Btn_REPORT.setMaximumWidth(200)
    horizontalLayout.addWidget(Push_Btn_REPORT)

    verticalLayout_3.addLayout(horizontalLayout)

    return PyTOP, verticalLayout_3, Tool_Btn_save_mean_relative_closeness, Push_Btn_RESET, Push_Btn_REPORT


def city_filtering_combo_block(PyTOP, verticalLayout_3, table_df):
    label_10_1 = QtWidgets.QLabel(PyTOP)
    label_10_1.setObjectName("label_10_1")
    verticalLayout_3.addWidget(label_10_1)
    Combo_Btn_city_filtering = QComboBox()
    Combo_Btn_city_filtering.setObjectName("Combo_Btn_city_filtering")

    LETTERS = "abcçdefgğhıi̇jklmnoöprsştuüvyz"
    unique_cities = sorted(set(flatten([elem.split(', ') for elem in table_df["Şehir"].drop_duplicates()])),
                           key=lambda i: LETTERS.index(i[0].lower()))
    unique_cities.insert(0, "All Cities")
    for city in unique_cities:
        Combo_Btn_city_filtering.addItem(city)

    verticalLayout_3.addWidget(Combo_Btn_city_filtering)

    return PyTOP, verticalLayout_3, label_10_1, Combo_Btn_city_filtering


def print_adjuster_report_combo_block(PyTOP, verticalLayout_3):
    label_11 = QtWidgets.QLabel(PyTOP)
    label_11.setObjectName("label_11")
    label_11.setFrameShape(QtWidgets.QFrame.Panel)
    verticalLayout_3.addWidget(label_11)

    return PyTOP, verticalLayout_3, label_11


def print_adjuster_report_button_block(PyTOP, verticalLayout_3, gridLayout):
    Push_Btn_print_adjuster_report = QtWidgets.QPushButton(PyTOP)
    font = QtGui.QFont()
    font.setFamily("Courier 10 Pitch")
    font.setBold(True)
    font.setItalic(False)
    font.setWeight(50)
    Push_Btn_print_adjuster_report.setFont(font)
    Push_Btn_print_adjuster_report.setObjectName(
        "Push_Btn_plot_adjuster_report")
    Push_Btn_print_adjuster_report.setStyleSheet("background-color: lightgray")
    Push_Btn_print_adjuster_report.setToolTip(
        "Creates the general performance reports of all adjusters and saves them in the selected folder. (THIS PROCESS IS LONG)")
    verticalLayout_3.addWidget(Push_Btn_print_adjuster_report)
    gridLayout.addLayout(verticalLayout_3, 1, 2, 1, 1)

    return PyTOP, verticalLayout_3, gridLayout, Push_Btn_print_adjuster_report


def draw_line(PyTOP, gridLayout):
    line = QtWidgets.QFrame(PyTOP)
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    line.setObjectName("line")
    gridLayout.addWidget(line, 1, 1, 2, 1)

    return PyTOP, gridLayout
