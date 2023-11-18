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
import numpy as np
import pandas as pd
import graphe as g
import signaux as s
from scipy.fft import fft, fftshift, fftfreq, ifft

# %% Constantes
PI = np.pi
COLOR = 'red'

# %% fft


def compute_fft(signal, Te):
    N = len(signal)
    out = fftshift(fft(signal))/N
    freq = fftshift(fftfreq(N, Te))
    return out, np.abs(out), freq


# %% Générations de signaux et fft
t = np.linspace(0, 1, 1000)
Te = t[1]-t[0]
signal = s.GenerateSinus(1, 20)(t)
#signal = .1+s.GenerateSinus(1, 100)(t)+s.GenerateSinus(3, 200)(t)+s.GenerateSinus(.5, 300)(t)+s.GenerateSinus(1, 500)(t)
#signal = s.GenerateSquare(1, 20)(t)
plt.plot(t, signal)
out, ampl, freq = compute_fft(signal, Te)
plt.plot(freq, ampl)


# %% Fichier 1 - Oscillo CSV
# Lecture du fichier et extraction des donénes utiles
df = pd.read_csv('bloc_3_data/B3_data_01.csv', header=2)
df = df.drop(df.index[-1])
df = df.drop(df.index[-1])
df = df.drop(df.index[-1])
time = df['Time(s)'].to_numpy()
data = df['Volt(V)'].to_numpy()

# Affichage du signal
style = {
    0: lambda ax: g.lin_XY(ax, title='Signal', x_label=r'Temps $t$', x_unit='s', y_label=r'Tension $V$', y_unit='V'),
    1: lambda ax: g.lin_XY(ax, x_label=r'Temps $t$', x_unit='s', y_label=r'Tension $V$', y_unit='V')
}
axs = g.new_mosaique(2, 1, style=style)
axs[0].plot(time, data, color=COLOR)
axs[1].plot(time[:150], data[:150], color=COLOR)
plt.show()

# Calcul de la fft
Te = time[1] - time[0]
_, ampl, freq = compute_fft(data, Te)
plt.plot(freq, ampl)

# Du blabla
board = g.empty(g.new_plot())
board.text(.05, .9, r"Si on a $s(t)=(1+km(t))p(t)$ avec:")
board.text(.1, .8, r"$m(t)=A_m \cos{\omega_m t}$")
board.text(.1, .7, r"$p(t)=A_p \cos{\omega_p t}$")
board.text(.05, .6, r"Alors $s(t)=(A_p+kA_mA_p\cos{\omega_mt})\cos{\omega_pt}$")
board.text(.05, .5,
           r"Donc $\tilde s(\nu)=\left(\frac{A_p}{2}+\frac{kA_mA_p}{4}[\delta(\nu-\frac{\omega_m}{2\pi})+\delta(\nu+\frac{\omega_m}{2\pi})]\right)\ast[\delta(\nu-\frac{\omega_p}{2\pi})+\delta(\nu+\frac{\omega_p}{2\pi})]$")
board.text(.05, .4,
           r"$i.e.$ $\tilde s(\nu)=\frac{A_p}{2}\left[\delta(\nu-\nu_p)+\delta(\nu+\nu_p)\right]+\frac{kA_mA_p}{4}\left[\delta\left(\nu-(\nu_p+\nu_m)\right)+\delta\left(\nu+(\nu_p+\nu_m)\right)+\delta\left(\nu-(\nu_p-\nu_m)\right)+\delta\left(\nu+(\nu_p-\nu_m)\right)\right]$")

# Fréquence de la porteuse
index_porteuse = np.argsort(-ampl)[0]
f_porteuse = freq[index_porteuse]

# Démodulation
new_data = data*s.GenerateSinus(1, f_porteuse)(time)
out, ampl, freq = compute_fft(new_data, Te)
ax = g.lin_XY(g.new_plot())
ax.plot(freq, ampl, COLOR)


def filtre(f, L):
    if -L/2 <= f <= L/2:
        return 1
    else:
        return 0


filtre = np.vectorize(filtre)
signal_demodule = ifft(filtre(freq, f_porteuse)*out)
ax = g.lin_XY(g.new_plot())
ax.plot(time[:150], data[:150])
ax.plot(time[:150], np.abs(signal_demodule)[:150])
