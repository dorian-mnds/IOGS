#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bibliothèque pour la transformée de fourier

Created on Tue Nov 14 2023

@author: Dorian Mendes
"""

import scipy.fft as sp
import numpy as np
import graphe as g


def fft(signal, Te):
    N = len(signal)
    out = sp.fft(signal, N)
    amplitudes = np.abs(sp.fftshift(out))/N
    frequencies = sp.fftshift(sp.fftfreq(N, Te))
    return amplitudes, frequencies


def plot_fft(amplitudes, frequencies, only_positive=False, *args, **kwargs):
    ax = g.new_plot()
    ax = g.lin_XY(ax,
                  title='Fourier Transform of the signal',
                  x_label=r'$f$', x_unit='Hz',
                  y_label='Amplitude')
    if only_positive:
        N = len(frequencies)
        frequencies = frequencies[N//2:]
        amplitudes = amplitudes[N//2:]
    ax.plot(frequencies, amplitudes, *args, **kwargs)


if __name__ == '__main__':
    import signaux as s
    ax = g.new_plot()
    ax = g.lin_XY(ax)
    t = np.linspace(0, 1, 500)
    y = s.GenerateSinus(1, 10)(t)+s.GenerateSinus(3, 30)(t)++s.GenerateSinus(2, 70)(t)
    ax.plot(t, y)
    out = fft(y, t[1]-t[0])
    plot_fft(*out, True, color='red')
