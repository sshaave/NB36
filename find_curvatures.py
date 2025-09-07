import numpy as np
from numpy import ndarray

from tverrsnitt import Tverrsnitt
from materialmodeller import CarbonMaterial, ConcreteMaterial, RebarMaterial


def find_curvatures(
    moments: ndarray | float,
    m_montasje: ndarray | float,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    length: float,
    rebar_material: RebarMaterial,
    tendon_material: RebarMaterial,
    carbon_material: CarbonMaterial,
    eps_cs: float,
    is_ck_not_cd: bool,
    print_fp: bool,
) -> ndarray:
    """Metode for å finne kurvaturer langs bjelkens. Antall snitt bestemt av len(moments)"""
    from tverrsnittsberegninger import find_equilibrium_strains
    from hjelpemetoder import find_eps_carbon

    # Starter med å gjøre om en evt float til ndarray
    if isinstance(moments, (float, int)):
        moments = np.array([moments])

    # Initialiserer verdier TODO! burde være innenfor løkka for å korrekte høyder
    kurvaturer: ndarray = np.zeros_like(moments, dtype=float)
    eps_ok, eps_uk = -0.0005, 0.0005
    height = tverrsnitt.get_height_i()


    forspenning: float = tendon_material.get_fp() if tendon_material is not None else 0 # forspenning i kN
    for i, (moment, m_i_montasje) in enumerate(zip(moments, m_montasje)):
        if eps_cs != 0:
            # Svinnmoment
            z_ok: float = height / 2 - tverrsnitt.get_d_top_avg()
            z_uk: float = tverrsnitt.get_d_bot_avg() - height / 2
            sum_ok_armering: float = tverrsnitt.get_a_top_sum()
            sum_uk_armering: float = tverrsnitt.get_a_bot_sum()

            # Enhet: N/mm2 * mm3 / 1000 = Nmm/1000 = Nm
            m_svinn: float = (
                rebar_material.get_e_s_rebar()
                * eps_cs
                * (z_ok * sum_ok_armering - z_uk * sum_uk_armering)
                / 1000
            )
        else:
            m_svinn = 0.

        # Summerer momenter
        m_i = moment * 1e3 + m_svinn
        m_i_m = m_i_montasje * 1e3 + m_svinn
        tverrsnitt.set_height_for_snitt(i)

        # Kompakte tverrsnittsbredder
        l_kompakt: float = tverrsnitt.get_kompakt_lengde()
        if l_kompakt != 0:
            length_i = length / (len(moments) - 1) * i
            if length_i <= l_kompakt or length_i >= length - l_kompakt:
                # I et område med kompakt tverrsnitt
                tverrsnitt.is_compact(True)
            else:
                tverrsnitt.is_compact(False)

        # print(f"Høyde i snitt {i}: {tverrsnitt.get_height_i():.1f} mm")
        if abs(m_i) < 0.1:
            kurvaturer[i] = 0
            continue
        if carbon_material is not None:
            # Gjør en kjøring pr iterasjon for å finne riktig initielltøyning i karbonfiber
            carbon_material.reset_0_state()
            eps_ok_0, eps_uk_0, _, _ = find_equilibrium_strains(
                m_i_m,
                material,
                tverrsnitt,
                rebar_material,
                tendon_material,
                carbon_material,
                eps_ok,
                eps_uk,
                is_ck_not_cd,
            )
            eps_carbon = find_eps_carbon(eps_ok_0, eps_uk_0, tverrsnitt)
            carbon_material.set_eps_s_0_state(eps_carbon)

        eps_ok, eps_uk, _, _ = find_equilibrium_strains(
            m_i,
            material,
            tverrsnitt,
            rebar_material,
            tendon_material,
            carbon_material,
            eps_ok,
            eps_uk,
            is_ck_not_cd,
        )

        # Sjekker om konvergens ble funnet
        
        if eps_ok == 0.0 and eps_uk == 0.0:
            # Ingen konvergens.
            #  Forblender (setter forspenning til 0 og prøver igjen)
            for j in [0.8, 0.6, 0.4, 0.2, 0.1, 0.0]:
                ny_forspenning = forspenning * j
                tendon_material.set_fp(ny_forspenning)
                eps_ok, eps_uk = -0.0005, 0.0005
                eps_ok, eps_uk, _, _ = find_equilibrium_strains(
                    m_i,
                    material,
                    tverrsnitt,
                    rebar_material,
                    tendon_material,
                    carbon_material,
                    eps_ok,
                    eps_uk,
                    is_ck_not_cd,
                )
                tendon_material.set_fp(forspenning)
                if eps_ok != 0.0 and eps_uk != 0.0:
                    if print_fp:
                        print(
                        f"Fant ikke konvergens i snitt {i} med moment {moment:.1f} kNm. Setter forspenning til {ny_forspenning:.0f} kN "
                    )
                    break

            if eps_ok == 0.0 and eps_uk == 0.0:
                raise ValueError(
                    f" ----- Fant ikke konvergens i snitt {i} med moment {moment} kNm, "
                    "selv etter å ha satt forspenning til 0. ----- "
                )
            # print(f"Fant konvergens etter å ha satt forspenning til 0. "
            #      f"eps_ok: {eps_ok:.7f}, eps_uk: {eps_uk:.7f}")

        kurvaturer[i] = (eps_ok - eps_uk) / height
        # print(f"Kurvatur {i}: {kurvaturer[i]:.12f}  (eps_ok: {eps_ok:.7f}, eps_uk: {eps_uk:.7f})")

    return kurvaturer
