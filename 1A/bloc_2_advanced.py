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
DPI = None  # For the save of images


# %% Pour afficher avec la notation scientifique
def as_si(x, ndp):
    """
    Create a type to formate a string into the scientific notation using LaTeX (in matplotlib)

    Parameters
    ----------
    x : float
        The number to convert into scientific notation.
    ndp : int
        The number of decimals.

    Returns
    -------
    str
        The formatted string in LaTeX.

    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


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
z = data[:, 0].astype('int')  # en µm
local_path = np.array(["bloc_2_data/"+x for x in data[:, 1]])
img_array = np.array(list(map(plt.imread, local_path)))  # Array of the images in the same order of z
dim = img_array[0].shape[0]  # The number of pixel. (here: 500)

pixel_range = np.arange(dim)
x, y = np.meshgrid(pixel_range, pixel_range)


# %% barycenter_arrayycenter
def barycenter_arrayycenter(img):
    """
    Compute the coordinates of the barycenter_arrayycenter of the image.

    Parameters
    ----------
    img : ndarray
        The image.

    Returns
    -------
    ndarray
        The coordinates of the barycenter_arrayycenter.
        The first element correspond to x and the second to y.

    """
    I_tot = np.sum(img)
    x_sum = np.sum(img, axis=0)
    y_sum = np.sum(img, axis=1)
    x = x_sum.dot(pixel_range)
    y = y_sum.dot(pixel_range)
    return np.array([x/I_tot, y/I_tot])


barycenter_array = np.array(list(map(barycenter_arrayycenter, img_array))).T  # Array with the barycenter_arrayycenter: axis0 -> x_barycenter_array | axis1 -> y_barycenter_array


# %% gaussian
def gaussian_2d(arg, A, x0, y0, w_x, w_y, theta, offset):
    x, y = arg
    a = (np.cos(theta)**2)/(2*w_x**2) + (np.sin(theta)**2)/(2*w_y**2)
    b = -(np.sin(2*theta))/(4*w_x**2) + (np.sin(2*theta))/(4*w_y**2)
    c = (np.sin(theta)**2)/(2*w_x**2) + (np.cos(theta)**2)/(2*w_y**2)
    z = offset + A*np.exp(-(a*((x-x0)**2)+2*b*(x-x0)*(y-y0)+c*((y-y0)**2)))
    return z.ravel()


# %% radius
def radius(z, w0, M):
    """
    Compute the radius of the beam.

    Parameters
    ----------
    z : ndarray
        The position of the beam's slice.
        (Unit: µm)
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
    return w0*np.sqrt(1+(z*M**2*WAVELENGH/(PI*w0**2))**2)*1e6


# %% Traitement des images
width_x = []
width_y = []

for i in range(len(z)):
    image = img_array[i]
    xb = barycenter_array[0, i]
    yb = barycenter_array[1, i]
    wx = 2*np.std(image[int(np.floor(xb)), :])
    wy = 2*np.std(image[:, int(np.floor(yb))])

    # On fait la méthode des moindre carrés
    args, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=[np.max(image)-np.min(image), xb, yb, wx, wy, 0, 0])
    A, x0, y0, w_x, w_y, theta, offset = args

    width_x.append(w_x*np.cos(theta))
    width_y.append(w_y*np.sin(theta))

    z = gaussian_2d((x, y), *args).reshape((dim, dim))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, image, cmap='viridis', alpha=.5)
    ax.plot_wireframe(x, y, z, rstride=7, cstride=7, linewidth=.5, color='k')
    plt.show(block=True)

width_x = np.array(width_x)*PIXEL
width_y = np.array(width_y)*PIXEL


# %% À la recherche de M^2
# On fit nos données avec le modèle.
args_x, _ = curve_fit(radius, z, width_x, p0=[width_x[0], 1])
w0_x, M_x = args_x
args_y, _ = curve_fit(radius, z, width_y, p0=[width_y[0], 1])
w0_y, M_y = args_y

# On crée une rampe pour l'affichage
z_min = np.min(z)
z_max = np.max(z)
z_plot = np.linspace(z_min, z_max)

# On crée un nouveau graphe
ax_M2_x = graphe.new_plot()
ax_M_x = graphe.lin_XY(
    ax_M2_x,
    x_label='$z$',
    x_unit='$\mu m$',
    y_label='$\omega_x(z)$',
    y_unit='$\mu m$',
    title=r"$\omega_{0,x}=%.2f \mu m$ et $M_x^2=%s$" % (w0_x*1e6, as_si(M_x**2, 2))
)

# On trace
ax_M2_x.plot(z_plot, radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.plot(z_plot, -radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.fill_between(z_plot, -radius(z_plot, w0_x, M_x), radius(z_plot, w0_x, M_x), color='r', alpha=.5)
ax_M2_x.scatter(z, width_x, color='k')
ax_M2_x.scatter(z, -width_x, color='k')

# On enregistre les graphes
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
ax_M2_y.plot(z_plot, radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.plot(z_plot, -radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.fill_between(z_plot, -radius(z_plot, w0_y, M_y), radius(z_plot, w0_y, M_y), color='r', alpha=.5)
ax_M2_y.scatter(z, width_x, color='k')
ax_M2_y.scatter(z, -width_x, color='k')
if DPI is not None:
    plt.savefig("bloc_2_export/waist_y.png", dpi=DPI)
plt.show(block=True)
