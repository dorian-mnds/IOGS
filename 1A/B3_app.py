#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Qt App

Created on Tue Nov 28 2023

@author: Dorian Mendes
"""
# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QSlider, QVBoxLayout, QLineEdit, QHBoxLayout, QCheckBox, QPushButton, QFileDialog
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

# %% Constantes
COLOR1 = '#4472c4'
COLOR2 = '#c55a11'
LW = 3
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

        self.setStyleSheet("background: #f2f2f2;")
        self.setWindowIcon(QIcon('IOGSlogo.jpg'))

        # Define Window title
        self.setWindowTitle("AM Demodulation")

        # self.setWindowIcon(QIcon('_inc/IOGS-LEnsE-logo.jpg'))
        self.setGeometry(50, 50, 1000, 700)

        # Main Layout
        self.main_widget = QWidget()
        self.grid = QGridLayout()
        self.main_widget.setLayout(self.grid)

        # Signal d'entrée
        self.win_graph_in = pg.GraphicsLayoutWidget()
        self.win_graph_in.setBackground("w")
        self.grid.addWidget(self.win_graph_in, 0, 0)
        self.graph_in = self.win_graph_in.addPlot(row=0, col=0, title="Signal d'entrée")
        self.graph_in.showGrid(x=True, y=True, alpha=1.0)
        self.graph_in.setLabel("bottom", "Time (s)")
        self.graph_in.setLabel("left", "Signal (u.a.)")
        self.curve_graph_in = self.graph_in.plot(pen=pg.mkPen(COLOR1, width=LW))

        # FFT du signal d'entrée
        self.fft_in = self.win_graph_in.addPlot(row=1, col=0, title="FFT du signal d'entrée")
        self.fft_in.showGrid(x=True, y=True, alpha=1.0)
        self.fft_in.setLabel("bottom", "Frequency (Hz)")
        self.fft_in.setLabel("left", "Amplitude (u.a.)")
        self.curve_fft_in = self.fft_in.plot(pen=pg.mkPen(COLOR1, width=LW))

        # Signal de sortie
        self.win_graph_out = pg.GraphicsLayoutWidget()
        self.win_graph_out.setBackground("w")
        self.grid.addWidget(self.win_graph_out, 0, 1)
        self.graph_out = self.win_graph_out.addPlot(row=0, col=0, title="Signal démodulé")
        self.graph_out.showGrid(x=True, y=True, alpha=1.0)
        self.graph_out.setLabel("bottom", "Time (s)")
        self.graph_out.setLabel("left", "Signal (u.a.)")
        self.curve_graph_out = self.graph_out.plot(pen=pg.mkPen(COLOR2, width=LW))

        # FFT du signal de sortie
        self.fft_out = self.win_graph_out.addPlot(row=1, col=0, title="FFT du signal x sinus à f_porteuse")
        self.fft_out.showGrid(x=True, y=True, alpha=1.0)
        self.fft_out.setLabel("bottom", "Frequency (Hz)")
        self.fft_out.setLabel("left", "Amplitude (u.a.)")
        self.curve_fft_out = self.fft_out.plot(pen=pg.mkPen(COLOR2, width=LW))

        self.setCentralWidget(self.main_widget)

        # Sélection de f_porteuse
        self.text_curseur_main_widget = QWidget()
        self.layout_text_curseur = QHBoxLayout()

        self.label_curseur_porteuse = QLabel("Fréquence de la porteuse:")
        self.input_curseur_porteuse = QLineEdit()
        self.input_curseur_porteuse.textChanged.connect(self.input_curseur_porteuse_change)
        self.unit_curseur_porteuse = QLabel("Hz")

        self.layout_text_curseur.addWidget(self.label_curseur_porteuse)
        self.layout_text_curseur.addWidget(self.input_curseur_porteuse)
        self.layout_text_curseur.addWidget(self.unit_curseur_porteuse)
        self.text_curseur_main_widget.setLayout(self.layout_text_curseur)

        self.curseur_porteuse_main_widget = QWidget()
        self.layout_curseur_porteuse = QVBoxLayout()

        self.curseur_main_widget = QWidget()
        self.layout_curseur = QHBoxLayout()

        self.curseur_porteuse = QSlider(Qt.Horizontal)
        self.curseur_porteuse.setMinimum(0)
        self.curseur_porteuse.setMaximum(100)
        self.curseur_porteuse.setTickPosition(QSlider.TicksBelow)
        self.curseur_porteuse.valueChanged.connect(self.curseur_porteuse_valuechange)

        self.button_freq = QPushButton('OK')
        self.button_freq.clicked.connect(self.freq_porteuse_changed)

        # Sélection fréquence du filtre
        self.select_filtre_freq_default = QCheckBox("=f_porteuse?")
        self.select_filtre_freq_default.clicked.connect(self.freq_filtre_changed)
        self.filtre_freq_label = QLabel("Largeur de la porte:")
        self.filtre_freq_input = QLineEdit()
        self.filtre_freq_input.textChanged.connect(self.freq_filtre_changed)
        self.filtre_freq_input.setText(str(0))
        self.filtre_freq_unit = QLabel('Hz')
        self.layout_filtre_freq = QHBoxLayout()
        self.select_freq_filtre = QWidget()

        self.layout_filtre_freq.addWidget(self.select_filtre_freq_default)
        self.layout_filtre_freq.addWidget(self.filtre_freq_label)
        self.layout_filtre_freq.addWidget(self.filtre_freq_input)
        self.layout_filtre_freq.addWidget(self.filtre_freq_unit)
        self.select_freq_filtre.setLayout(self.layout_filtre_freq)

        self.layout_curseur.addWidget(self.curseur_porteuse)
        self.layout_curseur.addWidget(self.button_freq)
        self.curseur_main_widget.setLayout(self.layout_curseur)
        self.layout_curseur_porteuse.addWidget(self.text_curseur_main_widget)
        self.layout_curseur_porteuse.addWidget(self.curseur_main_widget)
        self.layout_curseur_porteuse.addWidget(self.select_freq_filtre)
        self.curseur_porteuse_main_widget.setLayout(self.layout_curseur_porteuse)
        self.layout_curseur_porteuse.setAlignment(Qt.AlignTop)

        self.grid.addWidget(self.curseur_porteuse_main_widget, 1, 0)
        self.curseur_porteuse_main_widget.setStyleSheet("background-color: #4472c4;")

        self.plot_freq_porteuse = self.fft_in.plot(pen=pg.mkPen(COLOR2, width=LW))

        # Filtrage
        self.plot_filtre = self.fft_out.plot(pen=pg.mkPen(COLOR1, width=LW))

        # Sélection fichier, base d'encodage et enregistrement
        self.selection_widget = QWidget()
        self.layout_selection = QVBoxLayout()

        self.layout_encodage = QHBoxLayout()
        self.encodage_widget = QWidget()

        self.filepath_button = QPushButton("Select file")
        self.filepath_button.clicked.connect(self.select_file)
        self.name_selection = QLabel('')

        self.base_64_box = QCheckBox("Base 64")
        self.freq_fichier_label = QLabel("| Fréquence:")
        self.freq_fichier_input = QLineEdit()
        self.base_64_label = QLabel("kHz")
        self.input_button = QPushButton("OK")
        self.input_button.clicked.connect(self.get_data)

        self.layout_encodage.addWidget(self.filepath_button)
        self.layout_encodage.addWidget(self.base_64_box)
        self.layout_encodage.addWidget(self.freq_fichier_label)
        self.layout_encodage.addWidget(self.freq_fichier_input)
        self.layout_encodage.addWidget(self.base_64_label)
        self.layout_encodage.addWidget(self.input_button)
        self.encodage_widget.setLayout(self.layout_encodage)

        # -- Validation et enregistrement --
        self.layout_save = QHBoxLayout()
        self.save_widget = QWidget()
        self.play = QPushButton("Play")
        self.play.clicked.connect(self.play_song)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_data)

        # -- Sélection fichier --
        self.layout_export = QHBoxLayout()
        self.main_export_widget = QWidget()

        self.filepath_export_label = QLabel("Nom du fichier d'exportation:")
        self.export_file_path = QLineEdit()

        self.layout_export.addWidget(self.play)
        self.layout_export.addWidget(self.filepath_export_label)
        self.layout_export.addWidget(self.export_file_path)
        self.layout_export.addWidget(self.save_button)
        self.main_export_widget.setLayout(self.layout_export)

        # -- Layouts --
        self.layout_selection.addWidget(self.encodage_widget)
        self.selection_widget.setLayout(self.layout_selection)
        self.layout_selection.addWidget(self.name_selection)
        self.layout_selection.addWidget(self.main_export_widget)

        self.layout_selection.setAlignment(Qt.AlignTop)

        self.grid.addWidget(self.selection_widget, 1, 1)
        self.selection_widget.setStyleSheet("background-color: #c55a11;")

    def select_file(self):
        """
        Selection of the file corresponding of the signal to demodulate.

        Returns
        -------
        None.

        """
        path, _ = QFileDialog.getOpenFileName(None, "Select File", "")
        if path:
            self.path = path
            self.name_selection.setText('\t>>> '+path.split('/')[-1])
        else:
            print("No file selected.")

    def get_data(self):
        """
        Extract the data from the file and compute the fft of the inputing signal.

        Returns
        -------
        None.

        """
        if self.base_64_box.isChecked():
            chemin_fichier_base64 = self.path
            with open(chemin_fichier_base64, "rb") as fichier_enc:
                contenu_enc = fichier_enc.read()
                donnees_decodees = base64.b64decode(contenu_enc)
            self.signal_in = np.frombuffer(donnees_decodees, dtype=np.int16)
            self.fe = 1000*int(self.freq_fichier_input.text())  # Hz
            self.Te = 1/self.fe  # s
            self.time_in = np.array([0+k*self.Te for k in range(len(self.signal_in))])
        else:
            df = pd.read_csv(self.path, header=2)
            df = df.drop(df.index[-1])
            df = df.drop(df.index[-1])
            df = df.drop(df.index[-1])
            self.time_in = df['Time(s)'].to_numpy()
            self.Te = self.time_in[1]-self.time_in[0]
            self.signal_in = df['Volt(V)'].to_numpy()
            self.fe = 1/self.Te

        self.N = len(self.time_in)

        self.curve_graph_in.setData(self.time_in, self.signal_in)
        self.complex_fft_in, self.freq_in = fft(self.signal_in, self.Te)
        self.ampl_fft_in = np.abs(self.complex_fft_in)

        self.curve_fft_in.setData(self.freq_in[self.N//2:], self.ampl_fft_in[self.N//2:])

        self.curseur_porteuse.setMaximum(int(np.max(self.freq_in)))
        self.freq_porteuse = int(np.max(self.freq_in)/2)
        self.value_freq_filtre = self.freq_porteuse
        self.curseur_porteuse.setValue(self.freq_porteuse)
        self.input_curseur_porteuse.setText(str(int(np.max(self.freq_in)/4)))
        self.curseur_porteuse.setTickInterval(len(self.freq_in)//20)
        self.filtre_freq_input.setText(str(self.freq_porteuse))

    def input_curseur_porteuse_change(self):
        """
        Evolution of the input line of f_porteuse.

        Returns
        -------
        None.

        """
        txt = self.input_curseur_porteuse.text()
        if txt != '':
            self.freq_porteuse = int(txt)
        else:
            self.freq_porteuse = 0
        self.curseur_porteuse.setValue(self.freq_porteuse)

    def curseur_porteuse_valuechange(self):
        """
        Evolution of the cursor of f_porteuse.

        Returns
        -------
        None.

        """
        # Update de la valeur de la porteuse
        self.freq_porteuse = self.curseur_porteuse.value()
        self.input_curseur_porteuse.setText(str(self.freq_porteuse))
        self.plot_freq_porteuse.setData([self.freq_porteuse]*2, [0, 1.05*np.max(self.ampl_fft_in)])
        self.freq_filtre_changed()

    def freq_porteuse_changed(self):
        """
        Compute the FFT and the outputing signal.

        Returns
        -------
        None.

        """
        self.freq_filtre_changed()
        # Calcul du spectre après multiplication par un sinus à f_porteuse
        self.compute_fft_out, self.freq_after_multiplication = fft(self.signal_in*s.GenerateSinus(1, self.freq_porteuse)(self.time_in), self.Te)
        self.ampl_fft_after_multiplication = np.abs(self.compute_fft_out)
        self.curve_fft_out.setData(self.freq_after_multiplication[self.N//2:], self.ampl_fft_after_multiplication[self.N//2:])

        # Affichage du filtre passe-bas
        self.plot_filtre.setData(
            [0, self.value_freq_filtre/2, self.value_freq_filtre/2],
            [np.max(self.ampl_fft_after_multiplication), np.max(self.ampl_fft_after_multiplication), 0]
        )

        # Calcul du signal de sortie
        mask = (-self.value_freq_filtre/2 <= self.freq_after_multiplication) & (self.freq_after_multiplication <= self.value_freq_filtre/2)
        if self.base_64_box.isChecked():
            self.signal_out = np.real(ifft(mask*self.compute_fft_out)).astype(np.int16)
        else:
            self.signal_out = np.real(ifft(mask*self.compute_fft_out))
        self.curve_graph_out.setData(self.time_in, self.signal_out)

    def freq_filtre_changed(self):
        """
        Evolution of the input line of f_filtre

        Returns
        -------
        None.

        """
        if self.select_filtre_freq_default.isChecked():
            self.filtre_freq_input.setText(str(self.freq_porteuse))
        else:
            self.value_freq_filtre = int(self.filtre_freq_input.text())

    # def update_graphs(self):
    #     self.curve_graph_in.setData(self.time_in, self.signal_in)
    #     self.curve_graph_out.setData(self.time_in, self.signal_out)
    #     self.curve_fft_in.setData(self.freq_in[self.N//2:], self.ampl_fft_in[self.N//2:])
    #     self.curve_fft_out.setData(self.freq_after_multiplication[self.N//2:], self.ampl_fft_after_multiplication[self.N//2:])

    def save_data(self):
        """
        Save the outputing signal.

        Returns
        -------
        None.

        """
        name = self.export_file_path.text()
        if self.base_64_box.isChecked():
            wavfile.write("bloc_3_export/"+name+'.wav', self.fe, np.real(self.signal_out))
        else:
            np.savetxt('bloc_3_export/'+name+'.txt', np.stack([self.time_in, self.signal_out], axis=1), header="Time (s),Voltage (V)", delimiter=',', comments='')
        print("saved")

    def play_song(self):
        """
        Play the data.

        Returns
        -------
        None.

        """
        sd.play(self.signal_out/np.max(self.signal_out)*.5, self.fe)
        sd.wait()
        sd.stop()


# %% Éxécution
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
