#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Qt App

Created on Mon Dec 11 2023

@author: Dorian Mendes
"""

# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QSlider, QLineEdit, QCheckBox, QPushButton, QFileDialog
from Widgets.TimeFrequencyGraph import TimeFrequencyGraph
from Widgets.SelectionWidget import SelectionWidget

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

import pyqtgraph as pg
import numpy as np
from fourier_transform import fft, ifft
import signaux as s
import pandas as pd
import base64
from scipy.io import wavfile
import sounddevice as sd

# %% Constants
COLOR1 = '#4472c4'
COLOR2 = '#c55a11'
PI = np.pi

# %% Application


class MainWindow(QMainWindow):
    """ Creation of the window. """

    def __init__(self):
        """
        Initialisation of the window.

        Returns
        -------
        None.

        """
        super().__init__()

        # Define the color of the background of the window
        self.setStyleSheet("background: #f2f2f2;")

        # Define the title and the iconof the window
        self.setWindowTitle("AM Demodulation")
        self.setWindowIcon(QIcon('IOGSlogo.jpg'))

        # Define the geometry of the window
        self.setGeometry(50, 50, 1000, 700)

        # Main Layout
        self.main_widget = QWidget()
        self.grid = QGridLayout()
        self.main_widget.setLayout(self.grid)

        # Graphs
        self.in_graph = TimeFrequencyGraph('TIME IN', 'FREQ IN', COLOR1, COLOR2)
        self.grid.addWidget(self.in_graph, 0, 0)

        self.out_graph = TimeFrequencyGraph('TIME OUT', 'FREQ OUT', COLOR1, COLOR2)
        self.grid.addWidget(self.out_graph, 0, 1)

        # Selection
        self.select = SelectionWidget()
        self.grid.addWidget(self.select, 1, 1)

        # Set the main widget
        self.setCentralWidget(self.main_widget)

        # Tests:
        self.in_graph.setTimeData([0, 1, 2], [0, 1, 0])
        self.in_graph.setFrequencyData([0, 1, 2], [0, 1, 0])

        self.out_graph.setTimeData([0, 1, 2], [0, 1, 0])
        self.out_graph.setFrequencyData([0, 1, 2], [0, 1, 0])


# %% Éxécution
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
