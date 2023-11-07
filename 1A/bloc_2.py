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
DPI = None  # Pour l'enregistrement des images


# %% Extraction des images

# ---- Demandé par le sujet ----
# import csv
# z = []
# nom = []
# with open('bloc_2_data/data_bloc_2.csv', 'r') as f:
#     reader = csv.reader(f , delimiter=',')
#     for row in reader:
#         z = np.append(z,float(row[0]))
# local_path = np.array(["bloc_2_data/"+x for x in nom])
# img_array = np.array(list(map(plt.imread, local_path)))  # Array of the images in the same order of z
# dim = img_array[0].shape[0]

data = np.loadtxt("bloc_2_data/data_bloc_2.csv", delimiter=",", dtype=str)
z = data[:, 0].astype('float')  # en µm
local_path = np.array(["bloc_2_data/"+x for x in data[:, 1]])

# Array of the images in the same order of z
img_array = np.array(list(map(plt.imread, local_path)))
dim = img_array[0].shape[0]  # The number of pixel. (here: 500)

pixel_range = np.arange(dim)


# %% Barycenter
def barycenter(img):
    """
    Compute the coordinates of the barycenter of the image.

    Parameters
    ----------
    img : ndarray
        The image.

    Returns
    -------
    ndarray
        The coordinates of the barycenter.
        The first element correspond to x and the second to y.

    """
    I_tot = np.sum(img)
    x_sum = np.sum(img, axis=0)
    y_sum = np.sum(img, axis=1)
    x = x_sum.dot(pixel_range)
    y = y_sum.dot(pixel_range)
    return np.array([x/I_tot, y/I_tot])


# Array with the barycenter: axis0 -> x_barycenter_array | axis1 -> y_barycenter_array
barycenter_array = np.array(list(map(barycenter, img_array))).T


# %% gaussian
def gaussian(x, A, B, x0, w):
    """
    Gaussian function.
    A + B.exp(-2(x-x0)^2/w^2)

    Parameters
    ----------
    x : ndarray
        Variable of the function.
    A : float
        Offset of the gaussian.
    B : float
        Amplitude of the gaussian.
    x0 : float
        Center of the gaussian
    w : float
        The half-width of the gaussian

    Returns
    -------
    ndarray
        The evaluation of the variable array by the gaussian function.

    """
    return A+B*np.exp(-2*(x-x0)**2/w**2)


# %% radius
def radius(z, w0, M):
    """
    Compute the radius of the beam.

    Parameters
    ----------
    z : ndarray
        The position of the beam's slice.
        (Unit: mm)
    w0 : float
        The beam waist
        (Unit: µm)
    M : float
        The M^2 factor of the beam.

    Returns
    -------
    ndarray
        The radius array of the beam's slice corresponding to z.
        (Unit: µm)

    """
    return w0*np.sqrt(1+(z*1e-3*M**2*WAVELENGH/(PI*w0**2))**2)*1e6

# %% Second-moment width


def second_moment_width_X(image):
    xb = barycenter(image)[0]
    x_matrix = (pixel_range-xb)**2
    I_tot = np.sum(image)
    s_x = np.sum(image, axis=0).dot(x_matrix)
    return 4*np.sqrt(s_x/I_tot)


def second_moment_width_Y(image):
    yb = barycenter(image)[1]
    y_matrix = (pixel_range-yb)**2
    I_tot = np.sum(image)
    s_y = np.sum(image, axis=1).dot(y_matrix)
    return 4*np.sqrt(s_y/I_tot)


# %% Traitement des images
width_x = []
width_y = []

for i in range(len(z)):
    image = img_array[i]
    xb = barycenter_array[0, i]
    yb = barycenter_array[1, i]
    wx = np.std(image[int(xb), :])  # Approximation de la demi-largeur selon x
    wy = np.std(image[:, int(yb)])  # Approximation de la ldemi-argeur selon y

    # On fait la méthode des moindre carrés
    args_x, _ = curve_fit(
        gaussian,  # The fitting function
        pixel_range,  # [0, 1, 2, ..., dim-1]
        image[int(yb), :],
        p0=[0, np.max(image)-np.min(image), xb, wx]  # Offset, amplitude, valeur moyenne, largeur
    )
    args_y, _ = curve_fit(
        gaussian,
        pixel_range,
        image[:, int(xb)],
        p0=[0, np.max(image)-np.min(image), yb, wy]
    )
    A_x, B_x, x0, w_x = args_x
    A_y, B_y, y0, w_y = args_y

    width_x.append(w_x)
    width_y.append(w_y)

    # On affiche tout - Paramètres de la grille d'affichage
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

    # Level line at the width of the gaussian function.
    ax[1, 0].contour(image, [(np.max(image)-np.min(image)) /
                     np.exp(1)], colors=['k'], linestyles='dotted')

    # On affiche le profil selon x
    ax[1, 1].set_title(r"Coupe selon $y$")
    from_img_array, = ax[1, 1].plot(
        image[:, int(np.floor(xb))], pixel_range, c='r')  # On donne un nom pour sauver le style de la ligne et l'ajouter à la légende
    ax[1, 1].axhline(yb, c='red', ls=':')
    ax[1, 1].invert_yaxis()
    fitted, = ax[1, 1].plot(gaussian(pixel_range, *args_y),
                            pixel_range, c='green', lw=1, ls='--')  # On donne un nom pour sauver le style de la ligne et l'ajouter à la légende

    # On affiche le profil selon y
    ax[0, 0].set_title(r"Coupe selon $x$")
    ax[0, 0].plot(pixel_range, image[int(np.floor(yb)), :], c='r')
    ax[0, 0].axvline(xb, c='red', ls=':')
    ax[0, 0].plot(pixel_range, gaussian(
        pixel_range, *args_x), c='green', lw=1, ls='--')

    # On affiche quelques infos
    ax[0, 1].axis('off')
    ax[0, 1].set_title(r"$z={}$µm ({}/{})".format(z[i], i+1, len(z)))
    ax[0, 1].text(0, .6, f"Intensité minimale: {np.min(image)} ; Intensité maximale: {np.max(image)}")
    ax[0, 1].text(0, .5, f"Taille du faisceau en x:{2*w_x*PIXEL:.2f} µm")
    ax[0, 1].text(0, .4, f"Taille du faisceau en y:{2*w_y*PIXEL:.2f} µm")
    ax[0, 1].legend([from_img_array, fitted], [
                    "Donnée de l'image", "Modèle"], loc="upper center")

    # On ajuste la taille des graphes à l'image
    asp_x = np.diff(ax[0, 0].get_xlim())[0] / np.diff(ax[0, 0].get_ylim())[0]
    asp_x /= np.abs(np.diff(ax[1, 0].get_xlim())
                    [0] / np.diff(ax[1, 0].get_ylim())[0])
    ax[0, 0].set_aspect(asp_x)

    asp_y = np.abs(np.diff(ax[1, 1].get_xlim())[
                   0] / np.diff(ax[1, 1].get_ylim())[0])
    asp_y /= np.abs(np.diff(ax[1, 0].get_xlim())
                    [0] / np.diff(ax[1, 0].get_ylim())[0])
    ax[1, 1].set_aspect(asp_y)

    # On enregistre les graphes
    if DPI is not None:
        plt.savefig("bloc_2_export/analyse_image_"+str(i)+'.png', dpi=DPI)
    plt.show(block=True)

width_x = np.array(width_x)*PIXEL
width_y = np.array(width_y)*PIXEL

width_x_with_D4sigma = np.array(list(map(second_moment_width_X, img_array)))*PIXEL
width_y_with_D4sigma = np.array(list(map(second_moment_width_Y, img_array)))*PIXEL

# %% À la recherche de M^2
# On fit nos données avec le modèle.
args_x, _ = curve_fit(radius, z, width_x, p0=[width_x[0], 1])
w0_x, M_x = args_x
args_y, _ = curve_fit(radius, z, width_y, p0=[width_y[0], 1])
w0_y, M_y = args_y

# Avec D4sigma
args_x_D4, _ = curve_fit(radius, z, width_x_with_D4sigma, p0=[width_x[0], 1])
w0_x_D4, M_x_D4 = args_x_D4
args_y_D4, _ = curve_fit(radius, z, width_y_with_D4sigma, p0=[width_y[0], 1])
w0_y_D4, M_y_D4 = args_y_D4

# On crée une rampe pour l'affichage
z_min = np.min(z)
z_max = np.max(z)
z_plot = np.linspace(z_min, z_max)

# On crée un nouveau graphe
ax_M2_x = graphe.new_plot()
ax_M2_x = graphe.lin_XY(
    ax_M2_x,
    x_label='$z$',
    x_unit='$mm$',
    y_label='$\omega_x(z)$',
    y_unit='$\mu m$',
    title=r"$\omega_{0,x}=%.2f \mu m$ et $M_x^2=%.2f$" % (w0_x*1e6, M_x**2)
)

# On trace
ax_M2_x.plot(z_plot, radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.plot(z_plot, -radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.fill_between(z_plot, -radius(z_plot, w0_x, M_x),
                     radius(z_plot, w0_x, M_x), color='r', alpha=.5)
ax_M2_x.scatter(z, width_x, color='k', label='Fit gaussien')
ax_M2_x.scatter(z, -width_x, color='k')

ax_M2_x.plot(z_plot, radius(z_plot, w0_x_D4, M_x_D4), color='green')
ax_M2_x.plot(z_plot, -radius(z_plot, w0_x_D4, M_x_D4), color='green')
ax_M2_x.scatter(z, width_x_with_D4sigma, color='green', marker='x', label=r'D4$\sigma$')
ax_M2_x.scatter(z, -width_x_with_D4sigma, color='green', marker='x')
ax_M2_x.legend()

# On enregistre les graphes
if DPI is not None:
    plt.savefig("bloc_2_export/waist_x.png", dpi=DPI)
plt.show(block=True)

ax_M2_y = graphe.new_plot()
ax_M2_y = graphe.lin_XY(
    ax_M2_y,
    x_label='$z$',
    x_unit='$mm$',
    y_label='$\omega_y(z)$',
    y_unit='$\mu m$',
    title=r"$\omega_{0,y}=%.2f \mu m$ et $M_y^2=%.2f$" % (w0_y*1e6, M_y**2)
)
ax_M2_y.plot(z_plot, radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.plot(z_plot, -radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.fill_between(z_plot, -radius(z_plot, w0_y, M_y),
                     radius(z_plot, w0_y, M_y), color='r', alpha=.5)
ax_M2_y.scatter(z, width_x, color='k', label='Fit gaussien')
ax_M2_y.scatter(z, -width_x, color='k')

ax_M2_y.plot(z_plot, radius(z_plot, w0_y_D4, M_y_D4), color='green')
ax_M2_y.plot(z_plot, -radius(z_plot, w0_y_D4, M_y_D4), color='green')
ax_M2_y.scatter(z, width_y_with_D4sigma, color='green', marker='x', label=r'D4$\sigma$')
ax_M2_y.scatter(z, -width_y_with_D4sigma, color='green', marker='x')
ax_M2_y.legend()

if DPI is not None:
    plt.savefig("bloc_2_export/waist_y.png", dpi=DPI)
plt.show(block=True)

# %% Analyse critique des résultats
# print("\n\t\t ===== Analyse critique des résultats =====")
# print(
#     '--- Q1 ---',
#     "Si le fond n'est pas globalement à 0 alors un offset sera introduit à la fonction gaussienne.",
#     "Cependant, la fit intègre la différence MAX-MIN ce qui permet de gérer ces cas là.",
#     sep='\n', end='\n\n'
# )
# print(
#     '--- Q2 ---',
#     "Si l'ellipse n'est axée selon X et Y alors le programme actuel ne calculera pas les bonnes largeur et donc pas le bon facteur M2 (voir schéma).",
#     "Pour améliorer la méthode, il faudrait prendre en compte l'angle theta entre les axes de l'ellipse et les axes de l'image et appliqué le programme dans ce nouveau repère.",
#     sep='\n', end='\n\n'
# )
# plt.imshow(plt.imread('bloc_2_data/Lissajous.png'))
# plt.axis('off')
# plt.show()
