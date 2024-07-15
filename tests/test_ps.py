"""Test cases for the __main__ module."""

import numpy as np
from py21cmfast_tools import calculate_ps


def test_calculate_ps():
    test_lc = np.random.Generator(100 * 100 * 1000).reshape((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [5.0, 6.0, 10.0, 27.0]

    calculate_ps(
        test_lc,
        test_redshifts,
        HII_DIM=100,
        L=200,
        zs=zs,
        calc_2d=False,
        calc_1d=True,
        calc_global=True,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        HII_DIM=100,
        L=200,
        zs=zs,
        calc_1D=True,
        calc_global=True,
        interp=True,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        HII_DIM=100,
        L=200,
        zs=zs,
        calc_1D=True,
        calc_global=True,
        mu=0.5,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        HII_DIM=100,
        L=200,
        zs=zs,
        calc_1D=True,
        calc_global=True,
        interp=True,
        mu=0.5,
    )
