import math
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

    return alpha, sum_trykk, sum_f_strekk_armering, z

def objective_function_e_s(
    e_s,
    e_c,
    height,
    material,
    as_bot,
    d_bot,
    a_pre_bot,
    d_pre_bot,
    as_top,
    d_top,
    a_pre_top,
    d_pre_top,
    creep_eff,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """

    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_s, f_b, z = section_integrator(
        e_s, e_c, height, material, as_bot, as_top, d_bot, d_top,
        a_pre_bot, a_pre_top, d_pre_bot, d_pre_top, creep_eff
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = f_b * z

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_s / max(mom_b, 1e-6) - 1.0, alpha, mom_s, mom_b, z


def objective_function_e_c(
    e_s,
    e_c,
    height,
    material,
    as_bot,
    d_bot,
    a_pre_bot,
    d_pre_bot,
    as_top,
    d_top,
    a_pre_top,
    d_pre_top,
    creep_eff,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """

    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_s, f_b, z = section_integrator(
        e_s, e_c, height, material, as_bot, as_top, d_bot, d_top,
        a_pre_bot, a_pre_top, d_pre_bot, d_pre_top, creep_eff
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = f_b * z

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_b / max(mom_s, 1e-6) - 1.0, alpha, mom_b, z


def newton_optimize_e_s(
    e_c,
    height: ndarray,
    material,
    as_bot,
    as_top,
    d_bot,
    d_top,
    a_pre_bot,
    a_pre_top,
    d_pre_bot,
    d_pre_top,
    initial_guess,
    creep_eff,
):
    """
    Metode som gjør iterasjonen for e_s via Newton-Raphson.
    Inneholder egne sjekk for e_s-optimering og er derfor skilt fra e_c.
    """
    # Initialiserer
    max_iterations = 6
    tolerance = 1e-3
    e_s = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_s, mom_b, z = objective_function_e_s(
            e_s, e_c, height, material, as_bot, d_bot, a_pre_bot, d_pre_bot,
            as_top, d_top, a_pre_top, d_pre_top, creep_eff
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _, _ = objective_function_e_s(
            e_s + h, e_c, height, material, as_bot, d_bot, a_pre_bot, d_pre_bot,
            as_top, d_top, a_pre_top, d_pre_top, creep_eff
        )

        f_prime = (f_value2 - f_value) / h

        # Oppdaterer e_s basert på Newton-Raphson-metoden
        e_s -= step_size * f_value / f_prime

        # Sjekk om tøyningen er altfor stor
        if e_s > 0.016:
            if mom_s < mom_b:
                return -1.0, -1.0, -1.0, -1.0
            e_s = 0.016
            step_size *= 0.5  # Kan kanskje på sikt fjerne step_size.

        # Sjekk mot konvergenskriteriet
        if abs_f_value < tolerance:
            return e_s, alpha, mom_b, z
        elif e_s < 0.0 or math.isnan(e_s):
            e_s = 0.0001
            step_size *= 0.75

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0


def newton_optimize_e_c(
    e_s,
    height: ndarray,
    material,
    as_bot,
    as_top,
    d_bot,
    d_top,
    a_pre_bot,
    a_pre_top,
    d_pre_bot,
    d_pre_top,
    initial_guess,
    e_cu,
    creep_eff,
):
    """
    Metode som gjør iterasjonen for e_c via Newton-Raphson.
    Inneholder egne sjekk for e_c-optimering og er derfor skilt fra e_s.
    """
    # Initialiserer
    max_iterations = 6
    tolerance = 1e-3
    e_c = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_b, z = objective_function_e_c(
            e_s, e_c, height, material, as_bot, d_bot, a_pre_bot, d_pre_bot,
            as_top, d_top, a_pre_top, d_pre_top, creep_eff
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _ = objective_function_e_c(
            e_s, e_c + h, height, material, as_bot, d_bot, a_pre_bot, d_pre_bot,
            as_top, d_top, a_pre_top, d_pre_top, creep_eff
        )

        f_prime = (f_value2 - f_value) / h

        # Justerer verdi i henhold til Newton-Raphson-iterasjonen
        if abs_f_value > 1e-6:
            e_c -= step_size * f_value / f_prime

            # Sjekk om tøyningen er altfor stor
            if e_c > e_cu:
                e_c = e_cu
                step_size *= 0.5  # Kan kanskje på sikt fjerne step_size.

        # Sjekk mot konvergenskriteriet
        if abs_f_value < tolerance:
            return e_c, alpha, mom_b, z

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0

def integration_iterator_ultimate(height: ndarray, rebar_vector: ndarray, d_vector: ndarray, concrete_material: ConcreteMaterial, creep_eff: float) -> float:
    # Må ha en initiell testverdi
    initial_guess = 0.015

    # Kalkulasjon starter
    # Må finne den verdien/kombinasjonen av e_cu og e_s som gir indre likevekt i tverrsnittet.
    # Starter med å anta e_c = e_cu og ser om ulike verdier av e_s kan gi likevekt.

    sum_as_bot = sum(as_bot)
    sum_as_top = sum(as_top)

    if sum_as_bot > sum_as_top:
        e_s, alpha, mom_b, z = newton_optimize_e_s(
            e_c,
            height,
            material,
            as_bot,
            as_top,
            d_bot,
            d_top,
            a_pre_bot,
            a_pre_top,
            d_pre_bot,
            d_pre_top,
            initial_guess,
            creep_eff,
        )
    else:
        e_s = -1.0

    # Sjekker om første iterasjon var vellykket
    if e_s == -1.0:
        # Betongtøyningen kan ikke nå e_cu. Setter en armeringstøyning og finner
        # betongtøyning som gir likevekt i tverrsnittet (e_c < e_cu).
        e_s = initial_guess
        initial_guess = 0.00117
        e_c, alpha, mom_b, z = newton_optimize_e_c(
            e_s,
            height,
            material,
            as_bot,
            as_top,
            d_bot,
            d_top,
            a_pre_bot,
            a_pre_top,
            d_pre_bot,
            d_pre_top,
            initial_guess,
            e_cu,
            creep_eff,
    )

    return (alpha, mom_b, e_s, e_c, z)


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