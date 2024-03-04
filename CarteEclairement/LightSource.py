#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils Numériques pour l'Ingénieur.e en Physique.

Mini-Projet: Carte d'éclairement
>>> Modélisation d'une source lumineuse

Created on Tue Feb 27

@author: Dorian Mendes
"""

import numpy as np
from matplotlib import pyplot as plt
import graphe as g


class LightSource():
    """
    Modelisation of a light source.

    Attributes
    ----------
    instances : list
        The list of all the existing inctances of the class.
    x, y, z : float
        The coordinates of the light source in space.
    theta, phi : float
        The inclinaison of the source with respect to Z-axis.
    I0, FWHM : float
        The emission caracteristics

    Methods
    -------
    set_inclinaison
    set_intensity
    get_position
    get_direction

    """
    instances = []

    def __init__(self, x, y, z):
        # Ajoute la nouvelle instance que l'on crée dans 'instances'
        self.__class__.instances.append(self)

        self.x = x
        self.y = y
        self.z = z

        self.theta = 0
        self.phi = 0

    def set_inclinaison(self, theta, phi):
        """
        Set the orientation of the light source with respect to Z-axis.

        Parameters
        ----------
        theta : flaot
            Angle between X and Z-axis.
        phi : TYPE
            Angle between Y and Z-axis.

        Returns
        -------
        None.

        """
        self.theta = theta
        self.phi = phi

    def set_intensity(self, I0, theta):
        """
        Define the radiation indicator with the following model.

            I(alpha) = I0.exp(-4ln(2).(alpha/theta)^2)

        Parameters
        ----------
        I0 : float
            visual intensity forward on axis.
            unit: candela (Cd)
        theta : float
            The full width at half maximum.
            unit: degree (°)

        Returns
        -------
        None.

        """
        self.I0 = I0
        self.FWHM = theta

    def get_position(self):
        """
        Get the vector position of the source.

        Returns
        -------
        ndarray
            The vector postion (x, y, z).

        """
        return np.array([self.x, self.y, self.z])

    def get_direction(self):
        """
        Compute the normalized vector of the direction of the source.

        Returns
        -------
        ndarray
            Director vector of the axis of symmetry respecting the direction of lighting.

        """
        x_d = np.sin(np.deg2rad(self.theta))*np.cos(np.deg2rad(self.phi))
        y_d = np.sin(np.deg2rad(self.phi))*np.cos(np.deg2rad(self.theta))
        z_d = -np.cos(np.deg2rad(self.theta))
        return np.array([x_d, y_d, z_d])

    def get_intensity(self, obs_vect):
        """
        Compute the intensity at the observation point.

        Parameters
        ----------
        obs_vect : ndarray
            The observation vector.

        Returns
        -------
        float
            The intensity at the observation point.
            unit: candela (Cd)

        """
        # Vecteur from the center of the source to the observaztion point
        r_vect = obs_vect - self.get_position()

        # Normalization of r_vect
        u_vect = r_vect/np.linalg.norm(r_vect)

        # Angle between the symetry axis of the source and the observation point
        angle = np.arccos(self.get_direction().dot(u_vect))

        return self.I0*np.exp(-np.log(2)*(angle/self.FWHM)**2)


if __name__ == '__main__':
    src_1 = LightSource(0, 0, 4)
    src_1.set_inclinaison(0, 0)
    src_1.set_intensity(1, 30)

    # src_2 = LightSource(1, 1, 4)
    # src_2.set_inclinaison(90, 0)
    # src_2.set_intensity(5, 10)

    src_3 = LightSource(1, 1, 4)
    src_3.set_inclinaison(0, 90)
    src_3.set_intensity(5, 10)

    # src_4 = LightSource(1, 1, 4)
    # src_4.set_inclinaison(90, 90)
    # src_4.set_intensity(5, 10)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.set_zlim([0, 6])

    for light in LightSource.instances:
        ax.scatter(*light.get_position(), color='blue')
        ax.quiver(*light.get_position(), *light.get_direction(), color='blue')

    plt.show()
