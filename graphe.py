#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Bibliothèques de graphiques

Created on Tue Sep 26 2023

@author: Dorian Mendes
"""
# %% Libraries
import numpy as np
import matplotlib.pyplot as plt

# %% Graphes

class Lin_XY:
    def __init__(self):
        self.fig , self.ax = plt.subplots(1,1)

        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        
        self.ax.spines[['right','top']].set_color('none')
        
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        
        self.xlabel = '' ; self.ylabel = ''
        self.xunit = '' ; self.yunit = ''
        self.title = ''
        
    def update(self):
        if self.xunit == '':
            self.ax.set_xlabel(r"{}".format(self.xlabel) , loc='right')
        else:
            self.ax.set_xlabel(r"{} ({})".format(self.xlabel,self.xunit) , loc='right')
        if self.yunit == '':
            self.ax.set_ylabel(r"{}".format(self.ylabel) , loc='top')
        else:
            self.ax.set_ylabel(r"{} ({})".format(self.ylabel,self.yunit) , loc='top')
        self.ax.set_title(r"{}".format(self.title))
            
    def set_xlabel(self, xlabel):
        self.xlabel = xlabel
        self.update()
    
    def set_xunit(self, xunit):
        self.xunit = xunit
        self.update()
    
    def set_ylabel(self, ylabel):
        self.ylabel = ylabel
        self.update()

    def set_yunit(self, yunit):
        self.yunit = yunit
        self.update()
    
    def set_title(self, title):
        self.title = title
        self.update()
        
    def set_axis_interection(self,coord):
        x0,y0 = coord
        self.ax.spines['left'].set_position(['data',x0])
        self.ax.spines['bottom'].set_position(['data',y0])
        
    def plot(self, X, Y, *args, **kwargs):
        return self.ax.plot(X, Y, *args, **kwargs)
    
    def legend(self, *args, **kwargs):
        return self.ax.legend(*args, **kwargs)
    
    def grid(self, *args, **kwargs):
        return self.ax.grid(*args, **kwargs)
    
    def show(self):
        plt.show()