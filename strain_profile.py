from typing import Tuple
from numpy import ndarray
from tverrsnittsberegninger import section_integrator
from hjelpemetoder import eps_ok_uk_to_c_and_s
from materialmodeller import (
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
)
from tverrsnitt import Tverrsnitt

def nr_find_strain_profile(
    width: ndarray,
    height: ndarray,
    tverrsnitt: Tverrsnitt,
    concrete_material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    tendon_material: RebarMaterial,
    carbon_material: CarbonMaterial,
    f_ck_or_cd: int) -> Tuple[float, float, float, float, float, float, float]:
    """Returnerer eps_ok, eps_uk, alpha, mom_strekk, z, f_strekk, f_trykk"""

    max_iterations: int = 12
    tolerance: float = 1e-4
    delta: float = 1e-8
    f, alpha, f_strekk, f_trykk, z = 0.0, 0.0, 0.0, 0.0, 0.0
    bending: bool = True
    find_uk_vec: list[bool] = [True, False]
    
    # Bestemmer om det skal antas strekk i OK eller UK
    strekk_uk: bool = True
    
    # Setter maks tøyninger for bøyeformen
    max_eps_trykk, max_eps_strekk = concrete_material.get_eps_cu(), rebar_material.get_eps_s_u()
    if strekk_uk:
        max_eps_uk, max_eps_ok = max_eps_strekk, max_eps_trykk
    else:
        max_eps_uk, max_eps_ok = max_eps_trykk, max_eps_strekk
        
    for find_uk in find_uk_vec:
        if find_uk:
            delta_uk, delta_ok = delta, 0.0
        else:
            delta_uk, delta_ok = 0.0, delta
        
        eps_ok, eps_uk = max_eps_ok, max_eps_uk / 10.
        step_size: float = 1.0
        d_bot = tverrsnitt.get_d_bot()[0]
        d_top = tverrsnitt.get_d_top()[0]

        # Starter iterasjon
        for iteration in range(max_iterations):
            eps_ok, eps_s, _ = eps_ok_uk_to_c_and_s(
                eps_ok, eps_uk, height, strekk_uk, not strekk_uk,
                d_bot if find_uk else d_top, d_top if find_uk else d_bot
            )
            alpha, f_trykk, f_strekk, z = section_integrator(eps_ok, eps_s, tverrsnitt, concrete_material,
                rebar_material, tendon_material, carbon_material, f_ck_or_cd)
            
            f = max(f_strekk / (-f_trykk), 1e-6) - 1.0
            
            if abs(f) < tolerance:
                mom_strekk = f_strekk * z
                return eps_ok, eps_uk, alpha, mom_strekk, z, f_strekk, f_trykk
            
            eps_uk_d, eps_ok_d = eps_uk + delta_uk, eps_ok + delta_ok # TODO! Har jeg glemt noe her?

        
    eps_ok = 0.0
    eps_uk = 0.0
    alpha = 0.0
    mom_strekk = 0.0
    z = 0.0
    f_strekk = 0.0
    f_trykk = 0.0
    return eps_ok, eps_uk, alpha, mom_strekk, z, f_strekk, f_trykk

