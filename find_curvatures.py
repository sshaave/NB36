import numpy as np
from numpy import ndarray

from tverrsnitt import Tverrsnitt
from materialmodeller import CarbonMaterial, ConcreteMaterial, RebarMaterial


def find_curvatures(moments: ndarray | float, tverrsnitt: Tverrsnitt, material: ConcreteMaterial,
                    rebar_material: RebarMaterial = None, tendon_material: RebarMaterial = None,
                    carbon_material: CarbonMaterial = None, eps_cs: float = 0, creep_eff: float = 0,
                    is_ck_not_cd: bool = True) -> ndarray:
    """Metode for å finne kurvaturer langs bjelkens. Antall snitt bestemt av len(moments)"""
    from tverrsnittsberegninger import find_equilibrium_strains
    # Starter med å gjøre om en evt float til ndarray
    if isinstance(moments, (float, int)):
        moments = np.array([moments])

    # Initialiserer verdier
    kurvaturer: ndarray = np.zeros_like(moments, dtype=float)
    eps_ok, eps_uk = -0.0005, 0.0005
    tot_height = tverrsnitt.get_height_max()

    # Svinnmoment
    z_ok: float = tot_height / 2 - tverrsnitt.get_d_top_avg()
    z_uk: float = tverrsnitt.get_d_bot_avg() - tot_height / 2
    sum_ok_armering: float = tverrsnitt.get_a_top_sum()
    sum_uk_armering: float = tverrsnitt.get_a_bot_sum()
    forspenning: float = tendon_material.get_fp() if tendon_material is not None else 0
    print(f"Forspenning i tendon: {forspenning} kN")

    # Enhet: N/mm2 * mm3 / 1000 = Nmm/1000 = Nm
    m_svinn: float = rebar_material.get_e_s_rebar() * eps_cs * \
        (z_ok * sum_ok_armering - z_uk * sum_uk_armering) / 1000

    for i, moment in enumerate(moments):
        m_i = moment * 1e3 + m_svinn
        if abs(m_i) < 1e-4:
            kurvaturer[i] = 0
            continue
        eps_ok, eps_uk, _, _ = find_equilibrium_strains(
            m_i, material, tverrsnitt, rebar_material, tendon_material,
            carbon_material, creep_eff, eps_ok, eps_uk, is_ck_not_cd
        )
        
        # Sjekker om konvergens ble funnet
        if eps_ok == 0. and eps_uk == 0.:
            # Ingen konvergens. Forblender (setter forspenning til 0 og prøver igjen)
            print(f"Fant ikke konvergens i snitt {i} med moment {moment} kNm, ")
            tendon_material.set_fp(0)
            eps_ok, eps_uk = -0.0005, 0.0005
            eps_ok, eps_uk, _, _ = find_equilibrium_strains(
                m_i, material, tverrsnitt, rebar_material, tendon_material,
                carbon_material, creep_eff, eps_ok, eps_uk, is_ck_not_cd
            )
            tendon_material.set_fp(forspenning)
            print(f"forspenning: {tendon_material.get_fp()} kN")
            
            if eps_ok == 0. and eps_uk == 0.:
                raise ValueError(f" ----- Fant ikke konvergens i snitt {i} med moment {moment} kNm, "
                                 "selv etter å ha satt forspenning til 0. ----- ")
            print(f"Fant konvergens etter å ha satt forspenning til 0. "
                  f"eps_ok: {eps_ok:.7f}, eps_uk: {eps_uk:.7f}")
            
        kurvaturer[i] = (eps_ok - eps_uk) / tot_height
        #print(f"Kurvatur {i}: {kurvaturer[i]:.12f}  (eps_ok: {eps_ok:.7f}, eps_uk: {eps_uk:.7f})")


    return kurvaturer