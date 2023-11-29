#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Tests

Created on Tue Nov 28 2023

@author: Dorian Mendes
"""
# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout, QLineEdit, QHBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer

import pyqtgraph as pg
import numpy as np
from fourier_transform import fft, ifft
import signaux as s
import pandas as pd

# %% Constantes
COLOR1 = 'red'
COLOR2 = 'blue'
LW = 3
PI = np.pi

# %% Application


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # self.setWindowIcon(QIcon('IOGS-LEnsE-logo.jpg'))

        # Define Window title
        self.setWindowTitle("AM Demodulation")
        # self.setWindowIcon(QIcon('_inc/IOGS-LEnsE-logo.jpg'))
        self.setGeometry(50, 50, 1000, 700)

        # Main Layout
        self.main_widget = QWidget()
        self.grid = QGridLayout()
        # self.main_layout.setColumnStretch(0, 1)
        # self.main_layout.setColumnStretch(1, 4)
        self.main_widget.setLayout(self.grid)

        # Sélection du fichier à démoduler
        ...

        # Signal d'entrée
        self.win_graph_in = pg.GraphicsLayoutWidget()
        self.win_graph_in.setBackground("w")
        self.grid.addWidget(self.win_graph_in, 0, 0)
        self.graph_in = self.win_graph_in.addPlot(row=0, col=0, title="Signal d'entrée")
        self.graph_in.showGrid(x=True, y=True, alpha=1.0)
        self.curve_graph_in = self.graph_in.plot(pen=pg.mkPen(COLOR1, width=LW))

        # FFT du signal d'entrée
        self.fft_in = self.win_graph_in.addPlot(row=1, col=0, title="FFT du signal d'entrée")
        self.fft_in.showGrid(x=True, y=True, alpha=1.0)
        self.curve_fft_in = self.fft_in.plot(pen=pg.mkPen(COLOR1, width=LW))

        # Signal de sortie
        self.win_graph_out = pg.GraphicsLayoutWidget()
        self.win_graph_out.setBackground("w")
        self.grid.addWidget(self.win_graph_out, 0, 1)
        self.graph_out = self.win_graph_out.addPlot(row=0, col=0, title="Signal démodulé")
        self.graph_out.showGrid(x=True, y=True, alpha=1.0)
        self.curve_graph_out = self.graph_out.plot(pen=pg.mkPen(COLOR2, width=LW))

        # FFT du signal de sortie
        self.fft_out = self.win_graph_out.addPlot(row=1, col=0, title="FFT du signal x sinus à f_porteuse")
        self.fft_out.showGrid(x=True, y=True, alpha=1.0)
        self.curve_fft_out = self.fft_out.plot(pen=pg.mkPen(COLOR2, width=LW))

        self.setCentralWidget(self.main_widget)

        # Sélection de f_porteuse
        self.text_curseur_main_widget = QWidget()
        self.layout_text_curseur = QHBoxLayout()
        self.label_curseur_porteuse = QLabel("Fréquence de la porteuse:")
        self.input_curseur_porteuse = QLineEdit()
        self.unit_curseur_porteuse = QLabel("Hz")
        self.layout_text_curseur.addWidget(self.label_curseur_porteuse)
        self.layout_text_curseur.addWidget(self.input_curseur_porteuse)
        self.layout_text_curseur.addWidget(self.unit_curseur_porteuse)
        self.text_curseur_main_widget.setLayout(self.layout_text_curseur)
        self.input_curseur_porteuse.textChanged.connect(self.input_curseur_porteuse_change)

        self.curseur_porteuse_main_widget = QWidget()
        self.layout_curseur_porteuse = QVBoxLayout()

        self.curseur_porteuse = QSlider(Qt.Horizontal)
        self.curseur_porteuse.setMinimum(0)
        self.curseur_porteuse.setMaximum(100)
        self.curseur_porteuse.setTickPosition(QSlider.TicksBelow)
        self.curseur_porteuse.valueChanged.connect(self.curseur_porteuse_valuechange)

        self.layout_curseur_porteuse.addWidget(self.text_curseur_main_widget)
        self.layout_curseur_porteuse.addWidget(self.curseur_porteuse)
        self.curseur_porteuse_main_widget.setLayout(self.layout_curseur_porteuse)
        self.grid.addWidget(self.curseur_porteuse_main_widget, 1, 0)

        self.plot_freq_porteuse = self.fft_in.plot(pen=pg.mkPen(COLOR2, width=LW/2))

        # Filtrage
        self.plot_filtre = self.fft_out.plot(pen=pg.mkPen(COLOR1, width=LW/2))

        # Data test
        self.data_test()
        self.curve_graph_in.setData(self.time_in, self.signal_in)
        self.curve_fft_in.setData(self.freq_in, self.ampl_fft_in)

    def data_test(self):
        df = pd.read_csv('bloc_3_data/B3_data_01.csv', header=2)
        df = df.drop(df.index[-1])
        df = df.drop(df.index[-1])
        df = df.drop(df.index[-1])
        self.time_in = df['Time(s)'].to_numpy()
        self.Te = self.time_in[1]-self.time_in[0]
        self.signal_in = df['Volt(V)'].to_numpy()

        self.complex_fft_in, self.freq_in = fft(self.signal_in, self.Te)
        self.ampl_fft_in = np.abs(self.complex_fft_in)
        self.curseur_porteuse.setMaximum(int(np.max(self.freq_in)/2))
        self.curseur_porteuse.setValue(int(np.max(self.freq_in)/4))
        self.input_curseur_porteuse.setText(str(int(np.max(self.freq_in)/4)))
        self.curseur_porteuse.setTickInterval(len(self.freq_in)//20)

    def input_curseur_porteuse_change(self):
        txt = self.input_curseur_porteuse.text()
        if txt != '':
            self.freq_porteuse = int(txt)
        else:
            self.freq_porteuse = 0
        self.curseur_porteuse.setValue(self.freq_porteuse)

    def curseur_porteuse_valuechange(self):
        # Update de la valeur de la porteuse
        self.freq_porteuse = self.curseur_porteuse.value()
        self.input_curseur_porteuse.setText(str(self.freq_porteuse))
        self.plot_freq_porteuse.setData([self.freq_porteuse]*2, [0, 1.05*np.max(self.ampl_fft_in)])

        # Calcul du spectre après multiplication par un sinus à f_porteuse
        self.compute_fft_out, self.freq_after_multiplication = fft(self.signal_in*s.GenerateSinus(1, self.freq_porteuse)(self.time_in), self.Te)
        self.ampl_fft_after_multiplication = np.abs(self.compute_fft_out)
        self.curve_fft_out.setData(self.freq_after_multiplication, self.ampl_fft_after_multiplication)

        # Affichage du filtre passe-bas
        self.plot_filtre.setData(
            [-self.freq_porteuse/2, -self.freq_porteuse/2, self.freq_porteuse/2, self.freq_porteuse/2],
            [0, np.max(self.ampl_fft_after_multiplication), np.max(self.ampl_fft_after_multiplication), 0]
        )
        mask = (-self.freq_porteuse/2 <= self.freq_after_multiplication) & (self.freq_after_multiplication <= self.freq_porteuse/2)
        self.signal_out = ifft(mask*self.compute_fft_out)
        self.curve_graph_out.setData(self.time_in, np.real(self.signal_out))

    def compute_fft_in(self):
        pass

    def update_graphs(self):
        self.curve_graph_in.setData(self.time_in, self.signal_in)
        self.curve_graph_out.setData(self.time_out, self.signal_out)
        self.curve_fft_in.setData(self.freq_in, self.ampl_fft_in)
        self.curve_fft_out.setData(self.freq_after_multiplication, self.ampl_fft_after_multiplication)


# %% Éxécution
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
