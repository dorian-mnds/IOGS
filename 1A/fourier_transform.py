#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bibliothèque pour la transformée de fourier

Created on Tue Nov 14 2023

@author: Dorian Mendes
"""

# %% Bibliothèques
import matplotlib.pyplot as plt
import scipy.fft as sp
import numpy as np
import signaux as s
import graphe as g

# %% Constantes
COLOR = 'red'

# %% Functions


def fft(signal, Te):
    """
    Compute the FFT of the signal.

    Parameters
    ----------
    signal : ndarray
        Signal we want the FFT.
    Te : float
        Samphing time.

    Returns
    -------
    complex_fft : ndarray
        FFT of the signal.
    frequencies : TYPE
        Frequencies corresponding.

    """
    N = len(signal)
    complex_fft = sp.fftshift(sp.fft(signal, N))
    frequencies = sp.fftshift(sp.fftfreq(N, Te))
    return complex_fft, frequencies


def ifft(a, *args, **kwargs):
    # N = len(a)
    return sp.ifft(sp.fftshift(a, *args, **kwargs))


def irfft(a, *args, **kwargs):
    # N = len(a)
    return sp.irfft(sp.fftshift(a, *args, **kwargs))


# %% ~~ MAIN ~~~
if __name__ == '__main__':
    # Fonction sinusoïdale
    f = 100  # Hz
    time = np.linspace(0, 20/f, 500)
    N = len(time)
    data = s.GenerateSinus(3, f, phase=np.pi/4)(time)-1
    out, freq = fft(data, time[1]-time[0])

    plt.plot(time, data)
    plt.show()
    plt.plot(freq, np.abs(out)/N)
    plt.show()
    plt.plot(time, ifft(out))

    # Somme de fonctions sinusoïdales
    f1 = 30  # Hz
    f2 = 250  # Hz
    f3 = 400  # Hz
    time = np.linspace(0, 1000/f3, 40000)
    data = s.GenerateSinus(4, f1)(time)+s.GenerateSinus(6, f2)(time)+s.GenerateSinus(2, f3)(time)
    out, freq = fft(data, time[1]-time[0])

    ax = g.new_mosaique(2, 1, style={
        0: lambda ax: g.lin_XY(ax, title=r"Signal", x_label=r"$t$", x_unit='s', y_label=r"$s(t)$"),
        1: lambda ax: g.lin_XY(ax, title=r"FFT", x_label=r"$f$", x_unit='Hz', y_label=r"$\tilde s(f)$")})
    ax[0].plot(time[:1000], data[:1000], color=COLOR)
    ax[1].plot(freq[len(freq)//2-1500:len(freq)//2+1500], np.abs(out)[len(freq)//2-1500:len(freq)//2+1500], color=COLOR)
    ax[1].axvline(f1, ls=':', color='k')
    ax[1].axvline(-f1, ls=':', color='k')
    ax[1].axvline(f2, ls=':', color='k')
    ax[1].axvline(-f2, ls=':', color='k')
    ax[1].axvline(f3, ls=':', color='k')
    ax[1].axvline(-f3, ls=':', color='k')
    plt.show()

    # Somme de fonctions sinusoïdales
    f1 = 30  # Hz
    f2 = 250  # Hz
    time = np.linspace(0, 1000/f2, 40000)
    data = s.GenerateSinus(4, f1)(time)*s.GenerateSinus(6, f2)(time)
    out, freq = fft(data, time[1]-time[0])

    ax = g.new_mosaique(2, 1, style={
        0: lambda ax: g.lin_XY(ax, title=r"Signal", x_label=r"$t$", x_unit='s', y_label=r"$s(t)$"),
        1: lambda ax: g.lin_XY(ax, title=r"FFT", x_label=r"$f$", x_unit='Hz', y_label=r"$\tilde s(f)$")})
    ax[0].plot(time[:1000], data[:1000], color=COLOR)
    ax[1].plot(freq[len(freq)//2-1500:len(freq)//2+1500], np.abs(out)[len(freq)//2-1500:len(freq)//2+1500], color=COLOR)
    plt.show()

    # Fonction rectangle
    f = 100  # Hz
    time = np.linspace(0, 10/f, 4000)
    data = s.GenerateSquare(3, f)(time)-1
    out, freq = fft(data, time[1]-time[0])

    ax = g.new_mosaique(2, 1, style={
        0: lambda ax: g.lin_XY(ax, title=r"Signal", x_label=r"$t$", x_unit='s', y_label=r"$s(t)$"),
        1: lambda ax: g.lin_XY(ax, title=r"FFT", x_label=r"$f$", x_unit='Hz', y_label=r"$\tilde s(f)$")})
    ax[0].plot(time[:500], data[:500], color=COLOR)
    ax[1].plot(freq, np.abs(out), color=COLOR)
    plt.show()

    # Fonction gaussienne
    time = np.linspace(0, 50, 4000)
    data = np.exp(-(time-.6)**2/2*200)
    out, freq = fft(data, time[1]-time[0])

    ax = g.new_mosaique(2, 1, style={
        0: lambda ax: g.lin_XY(ax, title=r"Signal", x_label=r"$t$", x_unit='s', y_label=r"$s(t)$"),
        1: lambda ax: g.lin_XY(ax, title=r"FFT", x_label=r"$f$", x_unit='Hz', y_label=r"$\tilde s(f)$")})
    ax[0].plot(time[:100], data[:100], color=COLOR)
    ax[1].plot(freq[len(freq)//2-400:len(freq)//2+400], np.abs(out)[len(freq)//2-400:len(freq)//2+400], color=COLOR)
    plt.show()
