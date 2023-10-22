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
import graphe

# %% Constantes
PIXEL = 4.65  # µm - Taille d'un pixel
WAVELENGH = 1.3e-6  # m
PI = np.pi
DPI = None


# %% Pour afficher avec la notation scientifique
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


# %% Extraction des images
data = np.loadtxt("bloc_2_data/data_bloc_2.csv", delimiter=",", dtype=str)
z = data[:, 0].astype('int')  # en µm
path = np.array(["bloc_2_data/"+x for x in data[:, 1]])
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
    mosaique_style = {
        (0, 0): graphe.lin_XY,
        (1, 1): graphe.lin_XY,
        (0, 1): graphe.empty
    }
    fig, ax = graphe.new_mosaique(2, 2, style=mosaique_style, fig_output=True)
    fig.set_size_inches(7, 7)

    # On affiche l'image et le barycentre
    ax[1, 0].axis('off')
    ax[1, 0].imshow(image)
    ax[1, 0].axvline(xb, c='red', ls=':')
    ax[1, 0].axhline(yb, c='red', ls=':')
    ax[1, 0].scatter(xb, yb, c='red')
    ax[1, 0].contour(image, [np.max(image)/np.exp(1)], colors=['k'], linestyles='dotted')

    # On affiche le profil selon x
    ax[1, 1].set_title(r"Coupe selon $y$")
    from_img, = ax[1, 1].plot(image[:, int(np.floor(xb))], XX, c='r')
    ax[1, 1].axhline(yb, c='red', ls=':')
    ax[1, 1].invert_yaxis()
    fitted, = ax[1, 1].plot(gaussienne(XX, A_y, B_y, y0, w_y), XX, c='green', lw=1, ls='--')

    # On affiche le profil selon y
    ax[0, 0].set_title(r"Coupe selon $x$")
    ax[0, 0].plot(XX, image[int(np.floor(yb)), :], c='r')
    ax[0, 0].axvline(xb, c='red', ls=':')
    ax[0, 0].plot(XX, gaussienne(XX, A_x, B_x, x0, w_x), c='green', lw=1, ls='--')

    # On affiche quelques infos
    ax[0, 1].axis('off')
    ax[0, 1].set_title(r"$z={}$ ({}/{})".format(z[i], i+1, len(z)))
    ax[0, 1].text(0, .5, f"Taille du faisceau en x:{2*w_x*PIXEL:.2f} µm")
    ax[0, 1].text(0, .4, f"Taille du faisceau en y:{2*w_y*PIXEL:.2f} µm")
    ax[0, 1].legend([from_img, fitted], ["Donnée de l'image", "Modèle"], loc="upper center")

    # On ajuste la taille des graphes à l'image
    asp_x = np.diff(ax[0, 0].get_xlim())[0] / np.diff(ax[0, 0].get_ylim())[0]
    asp_x /= np.abs(np.diff(ax[1, 0].get_xlim())[0] / np.diff(ax[1, 0].get_ylim())[0])
    ax[0, 0].set_aspect(asp_x)

    asp_y = np.abs(np.diff(ax[1, 1].get_xlim())[0] / np.diff(ax[1, 1].get_ylim())[0])
    asp_y /= np.abs(np.diff(ax[1, 0].get_xlim())[0] / np.diff(ax[1, 0].get_ylim())[0])
    ax[1, 1].set_aspect(asp_y)

    width_x.append(w_x)
    width_y.append(w_y)

    if DPI is not None:
        plt.savefig("bloc_2_export/analyse_image_"+str(i)+'.png', dpi=DPI)
    plt.show(block=True)

width_x = np.array(width_x)*PIXEL
width_y = np.array(width_y)*PIXEL

# %% À la recherche de M^2
args_x, _ = curve_fit(rayon, z, width_x, p0=[width_x[0], 1])
w0_x, M_x = args_x
args_y, _ = curve_fit(rayon, z, width_y, p0=[width_y[0], 1])
w0_y, M_y = args_y

z_min = np.min(z)
z_max = np.max(z)
z_plot = np.linspace(z_min, z_max)

ax_M2_x = graphe.new_plot()
ax_M_x = graphe.lin_XY(
    ax_M2_x,
    x_label='$z$',
    x_unit='$\mu m$',
    y_label='$\omega_x(z)$',
    y_unit='$\mu m$',
    title=r"$\omega_{0,x}=%.2f \mu m$ et $M_x^2=%s$" % (w0_x*1e6, as_si(M_x**2, 2))
)
ax_M2_x.plot(z_plot, rayon(z_plot, w0_x, M_x), color='r')
ax_M2_x.plot(z_plot, -rayon(z_plot, w0_x, M_x), color='r')
ax_M2_x.fill_between(z_plot, -rayon(z_plot, w0_x, M_x), rayon(z_plot, w0_x, M_x), color='r', alpha=.5)
ax_M2_x.scatter(z, width_x, color='k')
ax_M2_x.scatter(z, -width_x, color='k')
if DPI is not None:
    plt.savefig("bloc_2_export/waist_x.png", dpi=DPI)
plt.show(block=True)

ax_M2_y = graphe.new_plot()
ax_M_y = graphe.lin_XY(
    ax_M2_y,
    x_label='$z$',
    x_unit='$\mu m$',
    y_label='$\omega_y(z)$',
    y_unit='$\mu m$',
    title=r"$\omega_{0,y}=%.2f \mu m$ et $M_y^2=%s$" % (w0_y*1e6, as_si(M_y**2, 2))
)
ax_M2_y.plot(z_plot, rayon(z_plot, w0_y, M_y), color='r')
ax_M2_y.plot(z_plot, -rayon(z_plot, w0_y, M_y), color='r')
ax_M2_y.fill_between(z_plot, -rayon(z_plot, w0_y, M_y), rayon(z_plot, w0_y, M_y), color='r', alpha=.5)
ax_M2_y.scatter(z, width_x, color='k')
ax_M2_y.scatter(z, -width_x, color='k')
if DPI is not None:
    plt.savefig("bloc_2_export/waist_y.png", dpi=DPI)
plt.show(block=True)
