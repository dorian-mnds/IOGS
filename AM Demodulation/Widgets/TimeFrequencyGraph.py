#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Qt App - Time/Frequency graph

Created on Mon Dec 11 2023

@author: Dorian Mendes
"""
# %% Librairies
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QVBoxLayout

import pyqtgraph as pg


# %% Widget
class TimeFrequencyGraph(QWidget):
    def __init__(self, time_title, freq_title, color1, color2, lw=3):
        super().__init__()

        self.graph = pg.GraphicsLayoutWidget()
        self.graph.setBackground("w")

        self.time_plot = self.graph.addPlot(row=0, col=0, title=time_title)
        self.time_plot.showGrid(x=True, y=True, alpha=1.0)
        self.time_plot.setLabel("bottom", "Time (s)")
        self.time_plot.setLabel("left", "Signal (u.a.)")
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(color1, width=lw))

        self.freq_plot = self.graph.addPlot(row=1, col=0, title=freq_title)
        self.freq_plot.showGrid(x=True, y=True, alpha=1.0)
        self.freq_plot.setLabel("bottom", "Frequency (Hz)")
        self.freq_plot.setLabel("left", "Amplitude (u.a.)")
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen(color2, width=lw))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.graph)
        self.setLayout(self.layout)

    def setTimeData(self, x_array, y_array):
        self.time_curve.setData(x_array, y_array)

    def setFrequencyData(self, x_array, y_array):
        self.freq_curve.setData(x_array, y_array)


# %% Launching as main for tests
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graph = TimeFrequencyGraph('Time plot test', 'Frequency plot test', COLOR1, COLOR2)

        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)
        self.layout.addWidget(self.graph)
        self.setCentralWidget(self.widget)

        self.graph.setTimeData([0, 1, 2], [0, 1, 0])
        self.graph.setFrequencyData([0, 1, 2], [0, 1, 0])


if __name__ == "__main__":
    COLOR1 = '#4472c4'
    COLOR2 = '#c55a11'

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
