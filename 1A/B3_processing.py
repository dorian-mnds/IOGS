#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Processing

Created on Tue Nov 24 2023

@author: Dorian Mendes
"""
# %% Bibliothèques
import matplotlib.pyplot as plt
import numpy as np
import graphe as g
import signaux as s
from fourier_transform import fft, ifft

# %% Constantes
COLOR1 = 'red'
COLOR2 = 'blue'


# %% Process
def demodulatation_AM_sinusoidal(time, data, name, slice=150):
    style = {
        (0, 0): lambda ax: g.lin_XY(ax, title=f'Signaux modulé ({COLOR1}) et démodulé ({COLOR2})', x_label=r'Temps $t$', x_unit='s', y_label=r'Tension $V$', y_unit='V'),
        (1, 0): lambda ax: g.lin_XY(ax, title="FFT du signal d'entrée", x_label=r'$f$', x_unit='Hz', y_label=r"$\left| \tilde{s}(f) \right|$", y_unit=r""),
        (0, 1): lambda ax: g.empty(ax),
        (1, 1): lambda ax: g.lin_XY(ax, title=r"FFT du signal d'entrée$\times\sin(2\pi f_p t)$", x_label=r'$f$', x_unit='Hz', y_label=r"$\left| \tilde{s}(f) \right|$", y_unit=r"")
    }
    axs = g.new_mosaique(2, 2, style, figsize=(10, 7))

    # Affichag des données
    axs[(0, 0)].plot(time[:slice], data[:slice], color=COLOR1, alpha=.5)

    # Calcul de la fft
    Te = time[1] - time[0]
    complex_fft, freq_fft = fft(data, Te)
    ampl_fft = np.abs(complex_fft)
    axs[(1, 0)].plot(freq_fft, ampl_fft, COLOR1)

    # Fréquence de la porteuse
    index_porteuse = np.argmax(np.abs(complex_fft[:len(complex_fft)//2]))
    f_porteuse = np.abs(freq_fft[index_porteuse])

    # Du texte
    axs[(0, 1)].text(.1, .9, f"L'affichage est tronqué à {slice} points.\nLa fréquence de la porteuse est {f_porteuse:.2e} Hz")

    # Démodulation
    new_data = data*s.GenerateSinus(1, f_porteuse)(time)
    complex_new_fft, freq_new_fft = fft(new_data, Te)
    ampl_new_fft = np.abs(complex_new_fft)
    axs[(1, 1)].plot(freq_new_fft, ampl_new_fft, COLOR2)

    mask = (-f_porteuse/2 <= freq_new_fft) & (freq_new_fft <= f_porteuse/2)
    axs[(1, 1)].plot(freq_new_fft, mask*np.max(ampl_new_fft), 'k:')
    complex_signal_demodule = ifft(mask*complex_new_fft)
    k = np.max(data)/np.max(np.abs(complex_signal_demodule))
    signal_demodule = k*np.abs(complex_signal_demodule)

    axs[(0, 0)].plot(time[:slice], signal_demodule[:slice], color=COLOR2)

    np.savetxt('bloc_3_export/'+name+'.csv', np.stack([time, signal_demodule], axis=1), header="Time (s),Voltage (V)", delimiter=',', comments='')
    return axs
