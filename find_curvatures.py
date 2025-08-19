def find_curvatures(moments: ndarray | float, tverrsnitt: Tverrsnitt, material: ConcreteMaterial,
                    rebar_material: RebarMaterial = None, tendon_material: RebarMaterial = None,
                    carbon_material: CarbonMaterial = None, eps_cs: float = 0, creep_eff: float = 0,
                    is_ck_not_cd: bool = True) -> ndarray:
    """Metode for å finne kurvaturer langs bjelkens. Antall snitt bestemt av len(moments)"""
    # Starter med å gjøre om en evt float til ndarray
    if isinstance(moments, (float, int)):
        moments = np.array([moments])

    # Initialiserer verdier
    kurvaturer: ndarray = np.zeros_like(moments)
    eps_ok, eps_uk = -0.0005, 0.0005
    tot_height = tverrsnitt.get_height_max()

    # Svinnmoment
    z_ok: float = tot_height / 2 - tverrsnitt.get_d_top_avg()
    z_uk: float = tverrsnitt.get_d_bot_avg() - tot_height / 2
    sum_ok_armering: float = tverrsnitt.get_a_top_sum()
    sum_uk_armering: float = tverrsnitt.get_a_bot_sum()

    m_svinn: float = rebar_material.get_e_s_rebar() * eps_cs * \
        (z_ok * sum_ok_armering - z_uk * sum_uk_armering) / 1000

    for i in range(moments):
        m_i = moments[i] * 1e3 + m_svinn
        if abs(m_i) < 1e-4:
            kurvaturer[i] = 0
            continue
        eps_ok, eps_uk, _, _ = find_equilibrium_strains(m_i, material, tverrsnitt, rebar_material, tendon_material,
                                                  carbon_material, creep_eff, eps_ok, eps_uk, is_ck_not_cd)
        kurvaturer[i] = (eps_ok - eps_uk) / tot_height


    return kurvaturer