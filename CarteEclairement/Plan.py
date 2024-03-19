#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils Numériques pour l'Ingénieur.e en Physique.

Mini-Projet: Carte d'éclairement
>>> Zone de travail

Created on Tue Mar 5

@authors:
    - Marion Bonvarlet
    - Dorian Mendes
"""

import numpy as np
from numpy import sin, cos, deg2rad
from matplotlib import pyplot as plt
# import graphe as g


class Plan():
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_lim = [x_min, x_max]
        self.y_lim = [y_min, y_max]

    def set_subdivisions(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

    def plot_radiance(self, lst_sources):
        X = np.linspace(self.x_lim[0], self.x_lim[1], self.Nx)
        Y = np.linspace(self.y_lim[0], self.y_lim[1], self.Ny)
        self.mat = np.zeros((self.Ny, self.Nx))

        for i in range(self.Nx):
            for j in range(self.Ny):
                r_obs = np.array([X[i], Y[j], 0])

                for light in lst_sources:
                    r_light = light.get_position()

                    d = np.linalg.norm(r_obs-r_light)
                    angle = np.arccos(light.get_direction().dot(r_obs-r_light)/d)
                    E = light.get_intensity(r_obs)*cos(angle)/(d**2)

                    self.mat[j, i] += E

        _X, _Y = np.meshgrid(X, Y)
        plt.pcolormesh(_X, _Y, self.mat)
        plt.colorbar()
        plt.set_cmap('gray')
        plt.axis('equal')


if __name__ == '__main__':
    from LightSource import LightSource
    Z = 2

    s1 = LightSource(1, 0, Z)
    s2 = LightSource(0, 1, Z)
    s3 = LightSource(0, -1, Z)
    s4 = LightSource(-1, 0, Z)

    s5 = LightSource(0, 0, 1)

    s1.set_inclinaison(45, 0)
    s2.set_inclinaison(0, 45)
    s3.set_inclinaison(0, -45)
    s4.set_inclinaison(-45, 0)
    s5.set_inclinaison(0, 0)

    s1.set_intensity(10, 60)
    s2.set_intensity(10, 60)
    s3.set_intensity(10, 60)
    s4.set_intensity(10, 60)
    s5.set_intensity(1, 20)

    plan = Plan(-5, 5, -5, 5)
    plan.set_subdivisions(100, 100)
    plan.plot_radiance(LightSource.instances)

    # # ----
    # plt.figure()
    # alpha = np.linspace(0, np.pi, 500)
    # plt.polar(alpha, np.exp(-(np.rad2deg(alpha)/20)**2), 'r')
