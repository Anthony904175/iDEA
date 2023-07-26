""" Contains all of the functionality required to perfrom an EDFT calculation """

## Importing helpful things ##
import warnings ## plan to remove after testing and development is done MAYBE
import numpy as np
import iDEA.utilities
import iDEA.interactions
import os
import copy
import string
import itertools
import functools
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import iDEA.system
import iDEA.state
import iDEA.methods.non_interacting


name = "edft"

def kinetic_energy_operator(s: iDEA.system.System) -> np.ndarray:
    r"""
    Computes the many particle KE operator as a matrix

    #### Will be very similar to the interacting and non interacting KE method

    Built using a given number of finite differences to represent the second derivative
    The number of differences is defined in s.stencil.

    | Args:
    |   s: system object

    | Returns:
    | K: sps.dia.matrix, KE operator
    """
    if s.stencil == 3:
        sd = 1.0 * np.array([1, -2, 1], dtype=float) / s.dx**2
        sdi = (-1, 0, 1)
    elif s.stencil == 5:
        sd = 1.0 / 12.0 * np.array([-1, 16, -30, 16, -1], dtype=float) / s.dx**2
        sdi = (-2, -1, 0, 1, 2)
    elif s.stencil == 7:
        sd = (
            1.0
            / 180.0
            * np.array([2, -27, 270, -490, 270, -27, 2], dtype=float)
            / s.dx**2
        )
        sdi = (-3, -2, -1, 0, 1, 2, 3)
    elif s.stencil == 9:
        sd = (
            1.0
            / 5040.0
            * np.array(
                [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], dtype=float
            )
            / s.dx**2
        )
        sdi = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    elif s.stencil == 11:
        sd = (
            1.0
            / 25200.0
            * np.array(
                [8, -125, 1000, -6000, 42000, -73766, 42000, -6000, 1000, -125, 8],
                dtype=float,
            )
            / s.dx**2
        )
        sdi = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    elif s.stencil == 13:
        sd = (
            1.0
            / 831600.0
            * np.array(
                [
                    -50,
                    864,
                    -7425,
                    44000,
                    -222750,
                    1425600,
                    -2480478,
                    1425600,
                    -222750,
                    44000,
                    -7425,
                    864,
                    -50,
                ],
                dtype=float,
            )
            / s.dx**2
        )
        sdi = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
    second_derivative = np.zeros((s.x.shape[0], s.x.shape[0]))
    for i in range(len(sdi)):
        second_derivative += np.diag(
            np.full(
                np.diag(np.zeros((s.x.shape[0], s.x.shape[0])), k=sdi[i]).shape[0],
                sd[i],
            ),
            k=sdi[i],
        )
    K = -0.5 * second_derivative
    return K

def external_potential_operator(s: iDEA.system.System) -> np.ndarray:
    r"""
    Compute the external potential operator.

    | Args:
    |     s: iDEA.system.System, System object.

    | Returns:
    |     Vext: np.ndarray, External potential energy operator.
    """
    Vext = np.diag(s.v_ext)
    return Vext





















##### STOPPING HERE TO FOCUS ON LEARNING THE THEORY BEHIND THE CALCULATIONS
##### READING AND TAKING NOTES ON THE GOK88 TRILLOGY
##### Will for sure need an exchange correlation potential and operatior
##### this is in the lda.py but might need to adjust it
##### Hamiltonian should be similar to the one in lda.py
##### Might want to use some of the python techniques used in lda.py
