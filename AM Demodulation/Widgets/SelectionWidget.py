#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Qt App - Selection widget

Created on Mon Dec 11 2023

@author: Dorian Mendes
"""
# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QCheckBox, QLineEdit

from PyQt5.QtCore import Qt


# %% Widget
class SelectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.super_layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.main_widget.setObjectName("main")
        self.main_layout = QVBoxLayout()

        # First line
        self.inport_file_data = QWidget()
        self.inport_file_layout = QHBoxLayout()

        self.select_file_button = QPushButton('Select file')
        self.base_64_box = QCheckBox('Base 64')
        self.sampling_freq_label = QLabel('| Fréquence:')
        self.sampling_freq_input = QLineEdit()
        self.sampling_freq_unit = QLabel('kHz')
        self.selct_validation_button = QPushButton('OK')

        self.inport_file_layout.addWidget(self.select_file_button)
        self.inport_file_layout.addWidget(self.base_64_box)
        self.inport_file_layout.addWidget(self.sampling_freq_label)
        self.inport_file_layout.addWidget(self.sampling_freq_input)
        self.inport_file_layout.addWidget(self.sampling_freq_unit)
        self.inport_file_layout.addWidget(self.selct_validation_button)

        self.inport_file_data.setLayout(self.inport_file_layout)
        self.main_layout.addWidget(self.inport_file_data)

        # Second line
        self.path_label = QLabel('')
        self.main_layout.addWidget(self.path_label)

        # Third line
        self.export_file_widget = QWidget()
        self.export_file_layout = QHBoxLayout()

        self.play_button = QPushButton('Play')
        self.export_label = QLabel('Export filename:')
        self.export_input = QLineEdit()
        self.export_validation_button = QPushButton('OK')

        self.export_file_layout.addWidget(self.play_button)
        self.export_file_layout.addWidget(self.export_label)
        self.export_file_layout.addWidget(self.export_input)
        self.export_file_layout.addWidget(self.export_validation_button)
        self.export_file_widget.setLayout(self.export_file_layout)
        self.main_layout.addWidget(self.export_file_widget)

        # Color of the entirely widget
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_widget.setLayout(self.main_layout)
        self.super_layout.addWidget(self.main_widget)
        self.setLayout(self.super_layout)
        style = """
            * {
                background-color: #c55a11;
            }
            #main {
                background-color: #c55a11;
                border: 2px solid black;
                border-radius:10px;
            }
            """
        self.setStyleSheet(style)

# %% Launching as main for tests


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selection = SelectionWidget()

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
