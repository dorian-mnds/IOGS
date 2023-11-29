#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Main

Created on Tue Nov 14 2023

@author: Dorian Mendes
"""

# %% Bibliothèques
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
from B3_processing import demodulatation_AM_sinusoidal, demodulatation_AM_son
from scipy.io import wavfile


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

# %% Fichier 2 - Fichier audio simple porteuse
# Lecture du fichier et extraction des donénes utiles
chemin_fichier_base64 = "bloc_3_data/B3_data_02.txt"
with open(chemin_fichier_base64, "rb") as fichier_enc:
    contenu_enc = fichier_enc.read()
    donnees_decodees = base64.b64decode(contenu_enc)
vecteur_donnees = np.array(list(donnees_decodees))
fe = 48000  # Hz
Te = 1/fe  # s
time = np.array([0+k*Te for k in range(len(vecteur_donnees))])
son = demodulatation_AM_son(time, vecteur_donnees, 'Fichier_2_audio', slice=len(time))
wavfile.write('bloc_3_export/Fichier_2.wav', fe, son)

# %% Fichier 2 fr - Fichier audio simple porteuse
# Lecture du fichier et extraction des donénes utiles
chemin_fichier_base64 = "bloc_3_data/B3_data_02_fr.txt"
with open(chemin_fichier_base64, "rb") as fichier_enc:
    contenu_enc = fichier_enc.read()
    donnees_decodees = base64.b64decode(contenu_enc)
vecteur_donnees = np.array(list(donnees_decodees))
fe = 48000  # Hz
Te = 1/fe  # s
time = np.array([0+k*Te for k in range(len(vecteur_donnees))])
son = demodulatation_AM_son(time, vecteur_donnees, 'Fichier_2_fr_audio', slice=len(time))
wavfile.write('bloc_3_export/Fichier_2_fr.wav', fe, son)
