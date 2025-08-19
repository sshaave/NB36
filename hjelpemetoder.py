"""Hjelpemetoder"""
import numpy as np
from numpy import ndarray
from typing import Tuple

from materialmodeller import CarbonMaterial, ConcreteMaterial, RebarMaterial
from strain_profile import find_curvatures
from tverrsnitt import Tverrsnitt

def eps_ok_uk_to_c_and_s(
    eps_ok: float,
    eps_uk: float,
    height: float,
    strekk_uk: bool,
    strekk_ok: bool,
    d0_strekk: float,
    d0_trykk: float
) -> Tuple[float, float, float]:
    """Metode for å gjøre om over- og underkantøyningter. Returerer betongtøyning, armeringstøyning og alpha."""
    tolerance: float = 1e-8
    if abs(eps_ok - eps_uk) < tolerance:
        return eps_ok, eps_ok, 0.0

    d_eps_dx: float = (eps_uk - eps_ok) / height

    if strekk_uk and not strekk_ok:
        # Strekk UK og trykk OK
        eps_s = eps_uk - d_eps_dx * (height - d0_strekk)
        alpha = -eps_ok / (eps_s - eps_ok)
        return eps_ok, eps_s, alpha

    if not strekk_uk and strekk_ok:
        # Strekk OK og trykk UK
        eps_s = eps_uk - d_eps_dx * (height - d0_strekk)
        alpha = -eps_uk / (eps_s - eps_uk)
        return eps_uk, eps_s, alpha

    if strekk_uk and strekk_ok:
        # Strekk i hele tverrsnittet
        eps_s_uk = eps_uk - d_eps_dx * (height - d0_strekk)
        eps_s_ok = eps_ok - d_eps_dx * (height - d0_trykk)
        return eps_s_ok, eps_s_uk, 0.0

    # else
    # Trykk i hele tverrsnittet
    if eps_ok < eps_uk:
        # Mest trykk i OK
        alpha_d = -eps_ok / d_eps_dx
    else:
        # Mest trykk i UK
        alpha_d = -eps_uk / d_eps_dx
    alpha = alpha_d / d0_strekk
    return min(eps_ok, eps_uk), 0.0, alpha
    

def calc_deflection_with_curvatures(moments: ndarray | float, lengde: float, tverrsnitt: Tverrsnitt,
                                    material: ConcreteMaterial, rebar_material: RebarMaterial = None,
                                    tendon_material: RebarMaterial = None,
                                    carbon_material: CarbonMaterial = None, eps_cs: float = 0, creep_eff: float = 0,
                                    is_ck_not_cd: bool = True) -> Tuple[ndarray, ndarray, ndarray, float]:
    """Metode for å regne ut forskyvning med krumninger."""
    curvatures = find_curvatures(moments, tverrsnitt, material, rebar_material, tendon_material,
                                 carbon_material, eps_cs, creep_eff, is_ck_not_cd)
    
    # Definerer iterasjonsverdier
    tolerance, max_iter = 1e-7, 15
    rotations, deflections = curvatures_to_deflections(curvatures, lengde, tolerance, max_iter)
    max_deflection = np.max(np.abs(deflections))
    
    return curvatures, rotations, deflections, max_deflection


def curvatures_to_deflections(curvatures: ndarray, lengde: float, tolerance: float, max_iter: int) -> Tuple[ndarray, ndarray]:
    """Konverterer krumninger til rotasjoner og forskyvninger."""
    n = len(curvatures)
    if n < 5:
        # egentlig 3 lol
        raise ValueError("Krumninger må ha minst 5 punkter for å kunne konverteres til rotasjoner og forskyvninger.") 
    
    # Antar jevn avstand mellom beregningssnittene
    dx = lengde / (n - 1)

    # Initialiserer vektorer
    rotations = np.zeros_like(curvatures)
    deflections = np.zeros_like(curvatures)

    # NR iterasjon med intiell forsøk på c. Må få 0 forskyvning i hver ende
    c_const = -0.0004
    
    for i in range(max_iter):
        # Rotasjoner
        raw_rotations = cumulative_trapezoidal(curvatures, dx)

        for i in range(n):
            rotations[i] = raw_rotations[i] + c_const

        # Deformasjoner
        raw_deflections = cumulative_trapezoidal(rotations, dx)
        deflections[:] = raw_deflections + 0  # direct assignment, no loop needed

        # Sjekk grenseverdier
        v_length = deflections[-1]

        if v_length.abs() < tolerance:
            break
    
        # Oppdatert c_const
        c_const -= v_length / lengde
    deflections[-1] = 0
    
    return rotations, deflections

def cumulative_trapezoidal(y: ndarray, dx: float) -> ndarray:
    """Integrerer opp"""
    n = len(y)
    integral = np.zeros_like(y)
    
    for i in range(1, n):
        area = 0.5 * (y[i - 1] + y[i]) * dx
        integral[i] = integral[i - 1] + area

    return integral

def help_function_grid(
    eps_ok: float,
    eps_uk: float,
    moment: float,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial = None,
    rebar_material: RebarMaterial = None,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
    creep_eff: float = 0,
    steps: int = 7,
    search_step: float = 0.000005
) -> Tuple[float, float]:
    from tverrsnittsberegninger import innerstate_beam
    """Hjelpemetode for å finne tøyninger ved hjelp av grid search."""
    # Implementasjon av grid search for å finne eps_ok og eps_uk
    best_norm = 1e12
    best_eps_ok = eps_ok
    best_eps_uk = eps_uk

    for d_ok in range(-steps, steps + 1):
        for d_uk in range(-steps, steps + 1):
            eps_ok_test = eps_ok + d_ok * search_step
            eps_uk_test = eps_uk + d_uk * search_step
            
            f_i, m_i = innerstate_beam(eps_ok_test, eps_uk_test, tverrsnitt, moment, material, rebar_material, rebar_pre_material, carbon_material, creep_eff, is_ck_not_cd)
            norm = np.sqrt(f_i**2 + m_i**2)
            
            if norm < best_norm:
                best_norm = norm
                best_eps_ok = eps_ok_test
                best_eps_uk = eps_uk_test
    
    return best_eps_ok, best_eps_uk

def help_function_steepest(eps_ok: float, eps_uk: float, moment: float,
                           tverrsnitt: Tverrsnitt, material: ConcreteMaterial = None,
                           rebar_material: RebarMaterial = None, rebar_pre_material: RebarMaterial = None,
                           carbon_material: CarbonMaterial = None, is_ck_not_cd: bool = True, creep_eff: float = 0,
                           steps=10) -> Tuple[float, float]:
    from tverrsnittsberegninger import innerstate_beam
    eps_ok_ = eps_ok
    eps_uk_ = eps_uk
    alpha = 0.001
    delta = 1e-6
    eps_cu_eff = material.get_eps_cu() * (1 + creep_eff) if is_ck_not_cd else material.get_eps_cu()
    if rebar_material is not None:
        eps_s_u: float = rebar_material.get_eps_s_u()
    elif rebar_pre_material is not None:
        eps_s_u: float = rebar_pre_material.get_eps_s_u()
    else:
        eps_s_u: float = 0.02
        print("Warning: No rebar material provided, using default eps_s_u = 0.02")
    
    prev_norm = float('inf')

    for _ in range(steps):
        f_i, m_i = innerstate_beam(eps_ok_, eps_uk_, tverrsnitt, moment, material, rebar_material, rebar_pre_material, carbon_material, creep_eff, is_ck_not_cd)
        norm = np.sqrt(f_i**2 + m_i**2)
        
        if norm > prev_norm:
            break
        prev_norm = norm
        
        # Numerical gradient
        f_ok, m_ok = innerstate_beam(eps_ok_ + delta, eps_uk_, tverrsnitt, moment, material, rebar_material, rebar_pre_material, carbon_material, creep_eff, is_ck_not_cd)
        f_uk, m_uk = innerstate_beam(eps_ok_, eps_uk_ + delta, tverrsnitt, moment, material, rebar_material, rebar_pre_material, carbon_material, creep_eff, is_ck_not_cd)
        
        grad_ok = (np.sqrt(f_ok**2 + m_ok**2) - norm) / delta
        grad_uk = (np.sqrt(f_uk**2 + m_uk**2) - norm) / delta

        grad_norm = np.sqrt(grad_ok**2 + grad_uk**2)
        eps_ok_ -= alpha * grad_ok / grad_norm
        eps_uk_ -= alpha * grad_uk / grad_norm
        
        # Clamp to physical range
        eps_ok_ = max(min(eps_ok_, eps_s_u), eps_cu_eff)
        eps_uk_ = max(min(eps_uk_, eps_s_u), eps_cu_eff)
    
    return eps_ok_, eps_uk_

def find_eps_carbon(eps_ok: float, eps_uk: float, tverrsnitt: Tverrsnitt) -> float:
    """Hjelpemetode for å finne tøyning i karbonfiber."""
    # Henter karbonfiberparametere
    a_carbon = tverrsnitt.get_a_carbon()
    d_carbon = tverrsnitt.get_d_carbon()
    
    if a_carbon.size == 0 or d_carbon.size == 0:
        return 0.0  # Ingen karbonfiber i tverrsnittet
    
    height_max = tverrsnitt.get_height_max()
    delta_eps: float = (eps_uk - eps_ok) / height_max
    eps_carbon = np.zeros_like(d_carbon, dtype=float)
    for i in range(d_carbon.size):
        eps_carbon[i] = eps_uk - delta_eps * (height_max - d_carbon[i])
    
    # Regn ut vektet snitt av karbonfiber
    sum_d_ganget_a = np.dot(eps_carbon, a_carbon).sum()
    snitt_eps_karbon = sum_d_ganget_a / a_carbon.sum()
    
    # Returnerer snitttøyning i karbonfiber
    if np.isnan(snitt_eps_karbon):
        print("Warning: NaN value encountered in carbon strain calculation.")
        return 0.0
    return snitt_eps_karbon