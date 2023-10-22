#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Bibliothèques de graphiques

Created on Tue Sep 26 2023

@author: Dorian Mendes
"""
# %% Libraries
import matplotlib.pyplot as plt

# %% New plot


def new_plot(fig_output=False):
    fig, ax = plt.subplots(1, 1)
    if fig_output:
        return fig, ax
    else:
        return ax


# %% New mosaique
def new_mosaique(nb_rows, nb_columns, style=None, fig_output=False):
    # Style sous la forme {(0,0): ..., (1,0):..., etc}
    fig, ax = plt.subplots(nb_rows, nb_columns, tight_layout=True)
    if style is not None:
        for k in style.keys():
            if k is not None:
                ax[k] = style[k](ax[k])
    if fig_output:
        return fig, ax
    else:
        return ax


# %% Axes modifiers
def lin_XY(ax, title='', x_label='', x_unit='', y_label='', y_unit='', axis_intersect=(0, 0)):
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines[['right', 'top']].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if x_unit == '':
        ax.set_xlabel(r"{}".format(x_label), loc='right')
    else:
        ax.set_xlabel(r"{} ({})".format(x_label, x_unit), loc='right')
    if y_unit == '':
        ax.set_ylabel(r"{}".format(y_label), loc='top')
    else:
        ax.set_ylabel(r"{} ({})".format(y_label, y_unit), loc='top')

    x0, y0 = axis_intersect
    ax.spines['left'].set_position(['data', x0])
    ax.spines['bottom'].set_position(['data', y0])

    ax.set_title(title)
    return ax


def empty(ax):
    ax.axis('off')
    return ax
