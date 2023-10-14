#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Évaluation bloc 1

Created on Tue Sep 26 2023

@author: Dorian Mendes
"""

# %% libraries
import numpy as np
from scipy.integrate import solve_ivp

import signaux
from graphe import Lin_XY


# %% Méthode d'Euler explicite
class Euler:
    """
    Solve a differential equation using Euler's method.

    Attributes
    ----------
    F : function
        The function representing the differential equation.

    Methods
    ----------
    solve
    """

    def __init__(self, F):
        """
        Parameters
        ----------
        F : function
            The function representing the differential equation: y'=F(t,y)
        """
        self.F = F

    def solve(self, time_range, init_conds, nb_points=100):
        """
        Solve the equation with Euler's method.

        Parameters
        ----------
        time_range : tuple
            A tuple with two elements: (t0, tf) where t0 represents the initial time and tf the final one.
        init_conds : ndarray
            An array with the same shape of the entry which represents the initials conditions of the equation.
        nb_points : int, optional
            The number of points used by the algorithm. (The default value is 100)

        Returns
        -------
        t : ndarray
            The array representing the time beetween t0 and tf.
        y: ndarray
            The array reprensenting the solution.
        """
        t0, tf = time_range
        delta_t = (tf-t0)/nb_points

        dim = init_conds.shape[0]

        t = np.zeros(nb_points)
        y = np.zeros((dim, nb_points))

        t[0] = t0
        y[:, 0] = init_conds

        for i in range(1, nb_points):
            t[i] = t[i-1] + delta_t
            y[:, i] = y[:, i-1] + delta_t * self.F(t[i-1], y[:, i-1])

        return np.array(t), np.array(y).T


# %% Circuit RC
class Circuit_RC:
    """
    Implemente the differential equation for a RC serie circuit.

    Attributes
    ----------
    R : float
        The resistance of the circuit (in Ohms)
    C : float
        The capacity of the circuit (in Farads)
    Ve : function
        The function of the input signal.
    Vs_0 : float
        The voltage for t=0s
    """

    def __init__(self, R, C, Ve, Vs_0):
        self.R = R
        self.C = C
        self.Ve = Ve
        self.Vs_0 = [Vs_0]
        self.tau = self.R * self.C

    def __call__(self, t, Vs):
        return (self.Ve(t)-Vs)/(self.tau)


# %% Circuit RC avec la méthode d'Euler
ve = signaux.GenerateConstant(value=5)
circuit = Circuit_RC(1e3, 1e-6, ve, Vs_0=0)
explicite_method = Euler(circuit)

# Création de la figure
graphe = Lin_XY()
graphe.set_title("Circuit RC - Méthode d'Euler")
graphe.set_xlabel('$t$')
graphe.set_xunit('s')
graphe.set_ylabel('$V_s$')
graphe.set_yunit('V')

# On trace la résolution pour des subdivisions de tailles différentes
for N in [10, 50, 100]:
    t_explicite, y_explicite = explicite_method.solve((0, 5*circuit.tau), np.array(circuit.Vs_0), N)
    graphe.plot(t_explicite, y_explicite, label=r"N={}".format(N))

# On trace la tension imposée, ici un échelon
graphe.plot(t_explicite, ve(t_explicite), label=r'$V_{IN}$', lw=.7, c='blue', ls='-.')

graphe.legend(loc=5)
graphe.grid()
graphe.show()

# %% Deuxième circuit RC avec solve_ivp

# Création de la figure
graphe_2 = Lin_XY()
graphe_2.set_title("Circuit RC - avec solve_ivp")
graphe_2.set_xlabel('$t$')
graphe_2.set_xunit('s')
graphe_2.set_ylabel('$V_s$')
graphe_2.set_yunit('V')

# On trace pour 3 fréquences différentes
for freq, color in [(10, 'blue'), (50, 'orange'), (100, 'green')]:
    ve = signaux.GenerateSinus(1, freq)
    circuit_2 = Circuit_RC(1e3, 1e-6, ve, 0)
    time = np.linspace(0, circuit_2.tau*5, 500)
    output_integr_RK45 = solve_ivp(circuit_2, [0, 5*circuit_2.tau], circuit_2.Vs_0, t_eval=time, method='RK45')

    graphe_2.plot(time, ve(time), ls=':', c=color, lw=.7)
    graphe_2.plot(output_integr_RK45.t, output_integr_RK45.y.T, label=r"$f={}$ Hz".format(freq), lw=.8, c=color)

graphe_2.legend(loc=8)
graphe_2.grid()
graphe_2.show()
