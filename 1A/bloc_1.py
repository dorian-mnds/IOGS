#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Bloc 1

Created on Tue Sep 26 2023

@author: Dorian Mendes
"""

# %% libraries
import numpy as np
from functools import wraps
from time import process_time
from scipy.integrate import solve_ivp

import signaux
from graphe import Lin_XY


# %% Timer
def timing(f):
    """
    Print the runtime of the function in argument

    Parameters
    ----------
    f : function
        The function we want to know the runtime.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ti = process_time()
        result = f(*args, **kw)
        tf = process_time()
        print(f"\tTemps d'éxécution de la fonction: {(tf-ti)*1e3:.2f} ms\n ******************************* \n")
        return result
    return wrap


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

    @timing
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


# %% Régime forcé
#ve = signaux.GenerateSinus(amplitude=5, frequency=10)
# ve = signaux.GenerateConstant(value=0)
# ve = signaux.GenerateSquare(amplitude=10, frequency=3)


# %%
class Circuit_RC:
    """
    Implemente the differential equation for a RC serie circuit.

    Attributes
    ----------
    R : float
        The resistance of the circuit
    C : float
        The capacity of the circuit
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


circ = Circuit_RC(100e3, 1e-6, ve, Vs_0=5)
explicite = Euler(circ)

# %%
graphe = Lin_XY()
graphe.set_title("Circuit RC")
graphe.set_xlabel('$t$')
graphe.set_xunit('s')
graphe.set_ylabel('$V_s$')
graphe.set_yunit('V')

for N in [10, 50, 500]:
    print(" ******************************* ")
    print(f"\t N={N}")

    t_explicite, y_explicite = explicite.solve((0, 5*circ.tau), np.array(circ.Vs_0), N)
    graphe.plot(t_explicite, y_explicite, label=r"N={}".format(N))

output_integr = solve_ivp(circ, [0, 5*circ.tau], circ.Vs_0, t_eval=t_explicite)
graphe.plot(output_integr.t, output_integr.y.T, label="With scipy", ls=':', c='black')

output_integr_RK23 = solve_ivp(circ, [0, 5*circ.tau], circ.Vs_0, t_eval=t_explicite)
graphe.plot(output_integr_RK23.t, output_integr_RK23.y.T, label="RK23", ls=':', c='red')

output_integr_RK45 = solve_ivp(circ, [0, 5*circ.tau], circ.Vs_0, t_eval=t_explicite)
graphe.plot(output_integr_RK45.t, output_integr_RK45.y.T, label="RK45", ls=':', c='green')

graphe.plot(t_explicite, ve(t_explicite), label='IN', lw=.7, c='blue', ls='-.')

graphe.legend()
graphe.grid()
graphe.show()


# %% Ordre 2 - Circuit RLC série
class Circuit_RLC:
    def __init__(self, R, L, C, Ve, init):
        self.R = R
        self.L = L
        self.C = C
        self.Ve = Ve
        self.init = init

    def __call__(self, t, y):
        Vs = y[0]
        u = y[1]
        return np.array([u, -self.R/self.L*u-1/(self.L*self.C)*(Vs-self.Ve(t))])


ve = signaux.GenerateSinus(4, 20)
rlc = Circuit_RLC(
    1e3,         # R
    1e-3,        # L
    10e-6,       # C
    ve,          # Signal d'entrée
    [1, 0])      # Vs(0) et u(0)
eq = Euler(rlc)

graphe_2 = Lin_XY()
graphe_2.set_title("Circuit RLC")

print(" ******************************* ")
t_rlc, y_rlc = eq.solve([0, .4], np.array([3, 0]))
graphe_2.plot(t_rlc, y_rlc.T[0], label=r'$V_S$')

output_integr_RK45 = solve_ivp(circ, [0, .4], rlc.init, t_eval=t_rlc)
graphe_2.plot(output_integr_RK45.t, output_integr_RK45.y[0], label="RK45", ls=':', c='green')

graphe_2.plot(t_rlc, ve(t_rlc), label='IN', lw=.7, c='blue', ls='-.')

graphe_2.legend()
graphe_2.show()
