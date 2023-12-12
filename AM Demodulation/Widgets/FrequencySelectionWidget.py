#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Qt App - Frequency Selection widget

Created on Tue Dec 12 2023

@author: Dorian Mendes
"""
# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QCheckBox, QLineEdit, QSlider

from PyQt5.QtCore import Qt

# %% Widget


class FrequencySelectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.super_layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.main_widget.setObjectName("main")
        self.main_layout = QVBoxLayout()

        # First line
        self.carrier_widget = QWidget()
        self.carrier_layout = QHBoxLayout()

        self.carrier_label = QLabel('Carrier frequency:')
        self.carrier_input = QLineEdit()
        self.carrier_unit = QLabel('Hz')

        self.carrier_layout.addWidget(self.carrier_label)
        self.carrier_layout.addWidget(self.carrier_input)
        self.carrier_layout.addWidget(self.carrier_unit)

        self.carrier_widget.setLayout(self.carrier_layout)
        self.main_layout.addWidget(self.carrier_widget)

        # Second line
        self.slider_widget = QWidget()
        self.slider_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider_validation = QPushButton('OK')

        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_validation)

        self.slider_widget.setLayout(self.slider_layout)
        self.main_layout.addWidget(self.slider_widget)

        # Third line

        # Color of the entirely widget
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_widget.setLayout(self.main_layout)
        self.super_layout.addWidget(self.main_widget)
        self.setLayout(self.super_layout)
        style = """
            * {
                background-color: #4472c4;
            }
            #main {
                background-color: #4472c4;
                border: 2px solid black;
                border-radius:10px;
            }
            """
        self.setStyleSheet(style)

# %% Launching as main for tests


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selection = FrequencySelectionWidget()

        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)
        self.layout.addWidget(QLabel('Test 1'))
        self.layout.addWidget(self.selection)
        self.layout.addWidget(QLabel('Test 2'))
        self.setCentralWidget(self.widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
