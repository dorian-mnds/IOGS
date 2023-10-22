#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour le Traitement de l'Information
Bloc 2

Created on Tue Oct 17

@author: Dorian Mendes
"""

# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %% Constantes
PIXEL = 4.65  # µm - Taille d'un pixel
WAVELENGH = 1.3e-6  # m
PI = np.pi

# %% Extraction des images
data = np.loadtxt("Data bloc 2/data_bloc_2.csv", delimiter=",", dtype=str)
z = data[:, 0].astype('int')  # en µm
path = np.array(["Data bloc 2/"+x for x in data[:, 1]])
img = np.array(list(map(plt.imread, path)))  # Array of the images in the same order of z
dim = img[0].shape[0]

XX = np.arange(dim)


# %% Barycentre
def barycentre(img):
    I_tot = np.sum(img)
    x_sum = np.sum(img, axis=0)
    y_sum = np.sum(img, axis=1)
    x = x_sum.dot(XX)
    y = y_sum.dot(XX)
    return np.array([x/I_tot, y/I_tot])


bar = np.array(list(map(barycentre, img))).T  # Array with the barycenter: axis0 -> x_bar | axis1 -> y_bar


# %% Gaussienne
def gaussienne(x, A, B, x0, w):
    return A+B*np.exp(-2*(x-x0)**2/w**2)


# %% Rayon
def rayon(z, w0, M):
    # z en µm , w(z) en µm
    return w0*np.sqrt(1+(z*M**2*WAVELENGH/(PI*w0**2))**2)*1e6


# %% Traitement des images
width_x = []
width_y = []

for i in range(len(z)):
    image = img[i]
    xb = bar[0, i]
    yb = bar[1, i]

    # On fait la méthode des moindre carrés
    args_x, _ = curve_fit(
        gaussienne, XX, image[int(np.floor(yb)), :],
        p0=[0, np.max(image), xb, 2*np.std(image[int(np.floor(xb)), :])]
    )
    args_y, _ = curve_fit(
        gaussienne, XX, image[:, int(np.floor(xb))],
        p0=[0, np.max(image), yb, 2*np.std(image[:, int(np.floor(yb))])]
    )
    A_x, B_x, x0, w_x = args_x
    A_y, B_y, y0, w_y = args_y

    # On affiche tout
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    # On affiche l'image et le barycentre
    ax[1, 0].axis('off')
    ax[1, 0].imshow(image)
    ax[1, 0].axvline(xb, c='red', ls=':')
    ax[1, 0].axhline(yb, c='red', ls=':')
    ax[1, 0].scatter(xb, yb, c='red')

    # On affiche le profil selon x
    ax[1, 1].plot(image[int(np.floor(yb)), :], XX, c='r')
    ax[1, 1].axhline(xb, c='red', ls=':')
    ax[1, 1].invert_yaxis()
    ax[1, 1].plot(gaussienne(XX, A_x, B_x, x0, w_x), XX, c='green', lw=1, ls='--')

    # On affiche le profil selon y
    ax[0, 0].plot(XX, image[:, int(np.floor(xb))], c='r')
    ax[0, 0].axvline(yb, c='red', ls=':')
    ax[0, 0].plot(XX, gaussienne(XX, A_y, B_y, y0, w_y), c='green', lw=1, ls='--')

    # On affiche quelques infos
    ax[0, 1].axis('off')
    ax[0, 1].text(0, 1, r"$z={}$ ({}/{})".format(z[i], i+1, len(z)))
    ax[0, 1].text(0, .9, f"Taille du faisceau en x:{2*w_x*PIXEL:.2f} µm")
    ax[0, 1].text(0, .8, f"Taille du faisceau en y:{2*w_y*PIXEL:.2f} µm")

    width_x.append(w_x)
    width_y.append(w_y)

    plt.show()

width_x = np.array(width_x)*PIXEL
width_y = np.array(width_y)*PIXEL

# %% À la recherche de M^2
average_width = (width_x+width_y)*.5
args, err = curve_fit(rayon, z, average_width, p0=[average_width[0], 1])
fig, ax = plt.subplots(1, 1)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines[['right', 'top']].set_color('none')
z_min = np.min(z)
z_max = np.max(z)
z_plot = np.linspace(z_min, z_max)
ax.plot(z_plot, rayon(z_plot, args[0], args[1]), color='r')
ax.plot(z_plot, -rayon(z_plot, args[0], args[1]), color='r')
ax.fill_between(z_plot, -rayon(z_plot, args[0], args[1]), rayon(z_plot, args[0], args[1]), color='r', alpha=.5)
ax.scatter(z, average_width, color='k')
ax.scatter(z, -average_width, color='k')
ax.set_xlabel(r'$z$ en $µm$', loc='right')
ax.set_ylabel(r'$w(z)$ en $µm$', loc='top')

ax.set_title(f"w0={args[0]*1e6:.2f} µm et M^2={args[1]**2:.2e}")
plt.show()
