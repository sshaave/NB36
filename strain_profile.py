import math
from typing import Tuple
import numpy as np
from numpy import array, ndarray
from materialmodeller import (
    CarbonFiber,
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    Material,
    RebarB400NC,
    RebarB500NC,
    Tendon,
)

def nr_find_strain_profile(
    width: ndarray,
    height: ndarray,
    concrete_material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    tendon_material: RebarMaterial,
    carbon_material: CarbonMaterial,
    creep: float,
    as_bot: ndarray,
    as_top: ndarray,
    d_bot: ndarray,
    d_top: ndarray,
    a_pre_bot: ndarray,
    a_pre_top: ndarray,
    d_pre_bot: ndarray,
    d_pre_top: ndarray,
    a_carbon: ndarray,
    d_carbon: ndarray,
    f_ck_or_cd: int) -> Tuple[float, float, float, float, float, float, float]:

    max_iterations: int = 12
    tolerance: float = 1e-4
    delta: float = 1e-8
    f, alpha, f_strekk, f_trykk, z = 0.0, 0.0, 0.0, 0.0, 0.0
    bending: bool = True
    find_uk_vec: list[bool] = [True, False]


    eps_ok = 0.0
    eps_uk = 0.0
    alpha = 0.0
    mom_strekk = 0.0
    z = 0.0
    f_strekk = 0.0
    f_trykk = 0.0
    return eps_ok, eps_uk, alpha, mom_strekk, z, f_strekk, f_trykk