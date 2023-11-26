#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 3 - Tests

Created on Tue Nov 24 2023

@author: Dorian Mendes
"""

# %% Bibliothèques
import signaux as s
import numpy as np
from numpy.random import rand, randint
from B3_processing import demodulatation_AM_sinusoidal


# %% Tests - Demodulation signal purement sinusoïdal
def demodulation_AM_sinusoidal_test(N):
    """
    Generate N random modulated signals and execute the demodulation function.

    Parameters
    ----------
    N : int
        The number of signals.

    Returns
    -------
    None.

    """
    for _ in range(N):
        f_m = randint(300, 700)
        f_p = randint(5000, 8000)
        A_m = rand()+.5
        A_p = 2*(rand()+.5)
        phi_m = rand()*2*np.pi
        K = 1.5*rand()
        offset = rand()*2-1

        time = np.array([0+k/(4*f_p) for k in range(4000)])

        signal_a_module = s.GenerateSinus(A_m, f_m, phi_m)(time)+offset
        porteuse = s.GenerateSinus(A_p, f_p)(time)
        signal_module = (K*signal_a_module+1)*porteuse

        axs = demodulatation_AM_sinusoidal(
            time, signal_module, f"test/sinusoidal_AM_fm_{f_m:.2f}_Am_{A_m:.2f}_phi_{phi_m:.2f}_offset_{offset:.2f}_fp_{f_p:.2f}_Ap_{A_p:.2f}_K_{K*100:.2f}%", slice=250)
        axs[(0, 0)].plot(time[:250], signal_a_module[:250], 'k')


# %% ~~ MAIN ~~~
if __name__ == '__main__':
    demodulation_AM_sinusoidal_test(10)
