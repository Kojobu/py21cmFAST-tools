"""Test cases for the __main__ module."""

import shutil

import numpy as np
import pytest
from typeguard import suppress_type_checks

from py21cmfast_tools import calculate_PS


def test_calculate_PS():
    test_lc = np.random.random(100*100*1000).reshape((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [5.,6.,10.,27.]

    out = calculate_PS(test_lc, test_redshifts, HII_DIM=100, L=200, zs=zs,
                       calc_1d = True, calc_global=True, calc_2d=False)
    
    out = calculate_PS(test_lc, test_redshifts, HII_DIM=100, L=200, zs=zs,
                       calc_1d = True, calc_global=True, interp=True)
    
    out = calculate_PS(test_lc, test_redshifts, HII_DIM=100, L=200, zs=zs,
                       calc_1d = True, calc_global=True, mu = 0.5)
    
    out = calculate_PS(test_lc, test_redshifts, HII_DIM=100, L=200, zs=zs,
                       calc_1d = True, calc_global=True, interp=True, mu = 0.5)
