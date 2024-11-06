from materialmodeller import CarbonMaterial, ConcreteMaterial, RebarMaterial, Material
import numpy as np

from abc import ABC, abstractmethod
from numpy import array, ndarray
from typing import List, Tuple, Iterator

def integrate_cross_section(e_ok: float, e_uk: float, height_uk: float, height: float, material, var_height: float) -> Tuple[float, float, float]:
    sum_f = 0
    sum_mom = 0

    iterations = 1000
    delta_e = (e_ok - e_uk) / iterations
    delta_h = height / iterations
    for i in range(iterations):
        height_i = height_uk + delta_h * i
        width_i = get_width(height_i, var_height)
        area_i = width_i * delta_h
        
        eps_i = e_uk + delta_e * i
        sigma_i = material.get_stress(eps_i)
        sum_f += area_i * sigma_i
        sum_mom += sum_f * height_i
    
    d = sum_mom / sum_f

    return sum_f, sum_mom, d


def evaluate_reinforcement_from_strain(d_vector: ndarray, a_vector: ndarray, d_0: float, e_c: float, e_s: float, creep_eff: float, material: Material, is_inside_concrete: bool) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    f_yd = material.get_f_yd()
    e_s_flyt = material.get_e_s_y()
    emod_s = material.get_E_s_rebar()  # N/mm2

    d_vec_strekk: ndarray = array()
    f_vec_strekk: ndarray = array()
    d_vec_trykk: ndarray = array()
    f_vec_trykk: ndarray = array()

    for d, as_ in zip(d_vector, a_vector):
        toyning = -e_c + (e_c - e_s) / d_0 * d
        if toyning >= 0:
            spenning = f_yd if toyning >= e_s_flyt else emod_s * toyning
        else:
            spenning = -f_yd if toyning <= -e_s_flyt else emod_s * toyning

        if toyning >= 0:
            # Tension
            d_vec_strekk.append(d)
            f_vec_strekk.append(spenning * as_)
        else:
            # Compression - using adjusted stress to account for displaced concrete area
            if is_inside_concrete:
                concrete_spenning = material.get_stress_at_strain(-toyning)
            else:
                concrete_spenning = 0
            justert_spenning = spenning + concrete_spenning
            d_vec_trykk.append(d)
            f_vec_trykk.append(justert_spenning * as_)

    return d_vec_strekk, f_vec_strekk, d_vec_trykk, f_vec_trykk


def section_integrator(e_ok: float, e_uk: float, height: float, material: ConcreteMaterial, rebar_vector: ndarray, d_vector: ndarray, creep: float, rebar_material: RebarMaterial, var_height: float, kryp: float, d_karbon: ndarray=None, carbon_vector: ndarray=None, carbon_material: CarbonMaterial=None):
    # Starter med å finne alpha fra tøyningene. Antar at tøyninger som gir trykk er positive, og strekk negativt. 
    delta_eps: float = (e_ok - e_uk) / height
    d_0 = d_vector[0]
    e_s_d0: float = e_uk + delta_eps * d_0
    alpha: float = e_ok / (e_ok - e_s_d0.abs)

    # Ønsker å finne hvilke lag som har strekk og trykk (og størrelse på kreftene)
    d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(d_vector, rebar_vector, d_0, e_ok, e_s_d0, kryp, rebar_material)
    if carbon_vector is not None:
        d_strekk_karbon, f_strekk_karbon, d_trykk_karbon, f_trykk_karbon = evaluate_reinforcement_from_strain(d_karbon, carbon_vector, d_0, e_ok, e_s_d0, 0, carbon_material)
    else:
        d_strekk_karbon: ndarray = array()
        f_strekk_karbon: ndarray = array()
        d_trykk_karbon: ndarray = array()
        f_trykk_karbon: ndarray = array()

    
    sum_f_strekk_armering = np.sum(f_strekk)
    sum_f_trykk_armering = np.sum(f_trykk)
    sum_f_strekk_karbon = np.sum(f_strekk_karbon)
    sum_f_trykk_karbon = np.sum(f_trykk_karbon)

    # Tyngdepunkt for armering
    d_strekk_avg: float = np.dot(f_strekk, d_strekk) / sum_f_strekk_armering
    alpha_d: float = alpha * d_0
    e_c_uk = 0. # bøyning, en del vil alltid være i strekk så setter denne 0 for integralet sin del

    height_uk = height - alpha_d
    f_bet, m_bet, d_bet = integrate_cross_section(e_ok, e_uk, height_uk, height, material, var_height)

    # Regner ut bidraget fra armering
    sum_trykkmoment: float = -np.dot(f_trykk, d_trykk) + m_bet
    sum_trykk: float = -sum_f_trykk_armering + f_bet
    d_trykk_avg: float = sum_trykkmoment / sum_trykk

    z = d_strekk_avg - d_trykk_avg

    return sum_trykk, sum_f_strekk_armering, z


def integration_iterator(height_vector: ndarray, width_vector: ndarray, 
                         rebar_vector: ndarray, d_vector: ndarray) -> float:
    # Må finne strekk og trykk for en gitt tøyning


    return 0

def get_width(height_i: float, var: float) -> float:
    """ Lager en funksjon som beskriver bredden for enhver høyde"""
    if height_i < 80:
        return 320
    elif height_i < 220: 
        return 320 - 220 * (height_i - 80) / 140
    elif height_i < 220 + var:
        return 100
    elif height_i < 220 + var + 50:
        rel_height = height_i - (220 + var)
        return 100 + 320 * (height_i - rel_height) / 50
    else: 
        return 420

if __name__ == "__main__":
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    sum_f, sum_m, d_bet = integrate_cross_section(0.0035, 0, 0, 200, betong_b35, 300)
    print(f"Force is {sum_f / 1000:.1f} kN")