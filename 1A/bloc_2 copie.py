#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils Numériques pour l'Ingénieur.e en Physique
Bloc 2

Created on Tue Oct 17

@author: Dorian Mendes
"""

# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import graphe

# %% Constantes
PIXEL = 4.65  # µm - Taille d'un pixel
WAVELENGH = 1.3e-6  # m
PI = np.pi
DPI = None  # For the save of images


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
        The half width of the gaussian

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


# %%
def get_slice(theta, x0, y0):
    if theta == np.deg2rad(90) or theta == np.deg2rad(-90):
        return [(x0, i) for i in range(dim)]
    XY = [(x0, y0)]
    k = 1
    while 0 <= int(y0+k*np.tan(theta)) < dim and 0 <= x0+k < dim:
        XY.append((int(x0+k), int(y0+k*np.tan(theta))))
        k += 1
    k = -1
    while 0 <= int(y0+k*np.tan(theta)) < dim and 0 <= x0+k < dim:
        XY.append((int(x0+k), int(y0+k*np.tan(theta))))
        k -= 1

    return sorted(XY)


def select_array(theta, i):
    image = img_array[i]
    xb = barycenter_array[0, i]
    yb = barycenter_array[1, i]
    XY = get_slice(theta, int(xb), int(yb))
    lst = []
    for x, y in XY:
        lst.append(image[y][x])
    return lst


def find_theta(theta, i):
    image = img_array[i]
    xb = barycenter_array[0, i]
    yb = barycenter_array[1, i]
    slce = select_array(theta, i)
    w = np.std(slce)*2
    return -w


# %% Traitement des images
width_u = []
width_v = []

for i in range(len(z)):
    image = img_array[i]
    arg_theta = minimize(find_theta, 0, i)
    theta = arg_theta.x[0]
    print(f"theta={theta}")
    slice_u = select_array(theta, i)
    slice_v = select_array(theta+PI/2, i)

    xb = barycenter_array[0, i]
    yb = barycenter_array[1, i]

    dim_u = len(slice_u)
    dim_v = len(slice_v)

    wu = np.std(slice_u)
    wv = np.std(slice_v)
    print(f"wu={wu} et wv={wv}")

    # On fait la méthode des moindre carrés
    args_u, _ = curve_fit(
        gaussian,  # The fitting function
        list(range(dim_u)),  # [0, 1, 2, ..., 255]
        slice_u,
        p0=[0, np.max(image)-np.min(image), xb, wu]
    )
    args_v, _ = curve_fit(
        gaussian,
        list(range(dim_v)),
        slice_v,
        p0=[0, np.max(image)-np.min(image), yb, wv]
    )
    A_u, B_u, xu, w_u = args_u
    A_v, B_v, yv, w_v = args_v

    width_u.append(w_u)
    width_v.append(w_v)

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
    from_img_array, = ax[1, 1].plot(slice_v, list(range(dim_v)), c='r')
    ax[1, 1].invert_yaxis()
    fitted, = ax[1, 1].plot(gaussian(pixel_range, *args_v),
                            pixel_range, c='green', lw=1, ls='--')

    # On affiche le profil selon y
    ax[0, 0].set_title(r"Coupe selon $x$")
    ax[0, 0].plot(list(range(dim_u)), slice_u, c='r')
    ax[0, 0].plot(pixel_range, gaussian(
        pixel_range, *args_u), c='green', lw=1, ls='--')

    # On affiche quelques infos
    ax[0, 1].axis('off')
    ax[0, 1].set_title(r"$z={}$ ({}/{})".format(z[i], i+1, len(z)))
    ax[0, 1].text(0, .5, f"Taille du faisceau en x:{2*w_u*PIXEL:.2f} µm")
    ax[0, 1].text(0, .4, f"Taille du faisceau en y:{2*w_v*PIXEL:.2f} µm")
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

width_u = np.array(width_u)*PIXEL
width_v = np.array(width_v)*PIXEL

# %% À la recherche de M^2
# On fit nos données avec le modèle.
args_x, _ = curve_fit(radius, z, width_u, p0=[width_u[0], 1])
w0_x, M_x = args_x
args_y, _ = curve_fit(radius, z, width_v, p0=[width_v[0], 1])
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
    title=r"$\omega_{0,x}=%.2f \mu m$ et $M_x^2=%.2f$" % (w0_x*1e6, M_x**2)
)

# On trace
ax_M2_x.plot(z_plot, radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.plot(z_plot, -radius(z_plot, w0_x, M_x), color='r')
ax_M2_x.fill_between(z_plot, -radius(z_plot, w0_x, M_x),
                     radius(z_plot, w0_x, M_x), color='r', alpha=.5)
ax_M2_x.scatter(z, width_u, color='k')
ax_M2_x.scatter(z, -width_u, color='k')

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
    title=r"$\omega_{0,y}=%.2f \mu m$ et $M_y^2=%.2f$" % (w0_y*1e6, M_y**2)
)
ax_M2_y.plot(z_plot, radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.plot(z_plot, -radius(z_plot, w0_y, M_y), color='r')
ax_M2_y.fill_between(z_plot, -radius(z_plot, w0_y, M_y),
                     radius(z_plot, w0_y, M_y), color='r', alpha=.5)
ax_M2_y.scatter(z, width_v, color='k')
ax_M2_y.scatter(z, -width_v, color='k')
if DPI is not None:
    plt.savefig("bloc_2_export/waist_y.png", dpi=DPI)
plt.show(block=True)
