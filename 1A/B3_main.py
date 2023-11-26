#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Main

Created on Tue Nov 14 2023

@author: Dorian Mendes
"""

# %% Bibliothèques
import numpy as np
import pandas as pd
import signaux as s
from B3_processing import *


# %% Fichier 1 - Oscillo CSV
# Lecture du fichier et extraction des donénes utiles
df = pd.read_csv('bloc_3_data/B3_data_01.csv', header=2)
df = df.drop(df.index[-1])
df = df.drop(df.index[-1])
df = df.drop(df.index[-1])
time = df['Time(s)'].to_numpy()
data = df['Volt(V)'].to_numpy()

# Démodulation
demodulatation_AM_sinusoidal(time, data, name='Oscillo_Demodulation')
plt.show()
