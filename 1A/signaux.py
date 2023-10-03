#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Bibliothèque de signaux

Created on Tue Sep 26 2023

@author: Dorian Mendes
"""
# %% Libraries
import numpy as np

# %% Constants
PI = np.pi

# %% Constant signal
class GenerateConstant:
    """
    Generate a constant signal.

    Attributes
    ----------
    value : float
        The amplitude of the signal
    """
    def __init__(self, value):
        """
        Parameters
        ----------
        value : float
            The amplitude of the signal
        """
        self.value = value
        
    def __call__(self, t):
        """
        Evaluate the signal at a time.

        Parameters
        ----------
        t : ndarray or float
            The array reprensenting the time.
        
        Returns
        -------
        ndarray or float
            An array representing the signal evaluates for thoses times.
        """
        return self.value * np.ones_like(t)

# %% Sinusoidal signal
class GenerateSinus:
    """
    Generate a sinusoidal signal.

    Attributes
    ----------
    amplitude : float
        The amplitude of the signal
    frequency : float
        The frequency of the signal
    """
    def __init__(self, amplitude, frequency):
        """
        Parameters
        ----------
        amplitude : float
            The amplitude of the signal
        frequency : float
            The frequency of the signal
        """
        self.amplitude = amplitude
        self.frequency = frequency
        
    def __call__(self, t):
        """
        Evaluate the signal at a time.

        Parameters
        ----------
        t : ndarray or float
            The array reprensenting the time.
        
        Returns
        -------
        ndarray or float
            An array representing the signal evaluates for thoses times.
        """
        return self.amplitude*np.sin(2*PI*self.frequency*t)
    
# %% Square signal
class GenerateSquare:
    """
    Generate a square signal.

    Attributes
    ----------
    amplitude : float
        The amplitude of the signal
    frequency : float
        The frequency of the signal
    symetry : float, optional
        ...
    """
    def __init__(self, amplitude, frequency, symetry=.5):
        """
        Parameters
        ----------
        amplitude : float
            The amplitude of the signal
        frequency : float
            The frequency of the signal
        symetry : float, optional
            ...
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.symetry = symetry
        
    def __call__(self, t):
        """
        Evaluate the signal at a time.

        Parameters
        ----------
        t : ndarray or float
            The array reprensenting the time.
        
        Returns
        -------
        ndarray or float
            An array representing the signal evaluates for thoses times.
        """
        T = 1/self.frequency
        def f(t):
            if 0 <= t%T <= self.symetry*T:
                return self.amplitude
            else:
                return -self.amplitude
        f = np.vectorize(f)
        return f(t)
    
# %% Triangle signal
class GenerateTriangle:
    """
    Generate a triangle signal.

    Attributes
    ----------
    amplitude : float
        The amplitude of the signal
    frequency : float
        The frequency of the signal
    symetry : float, optional
        ...
    """
    def __init__(self, amplitude, frequency, symetry=.5):
        """
        Parameters
        ----------
        amplitude : float
            The amplitude of the signal
        frequency : float
            The frequency of the signal
        symetry : float, optional
            ...
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.symetry = symetry
        
    def __call__(self, t):
        """
        Evaluate the signal at a time.

        Parameters
        ----------
        t : ndarray or float
            The array reprensenting the time.
        
        Returns
        -------
        ndarray or float
            An array representing the signal evaluates for thoses times.
        """
        T = 1/self.frequency
        def f(t):
            if 0 <= t%T <= self.symetry*T:
                return self.amplitude/(self.symetry*T)*(t%T)
            else:
                return self.amplitude+self.amplitude/((self.symetry-1)*T)*(t%T-self.symetry*T)
        f = np.vectorize(f)
        return f(t)