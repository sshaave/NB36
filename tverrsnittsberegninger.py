"""Tverrsnittsberegninger for karbonfiberforsterket spennarmert betongtverrsnitt"""

import math
import sys
from typing import Tuple
import numpy as np
from numpy import array, ndarray
from materialmodeller import (
    CarbonFiber,
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    Material,
    Tendon,
)
from hjelpemetoder import eps_c_and_eps_s_to_eps_ok_uk, help_function_grid, help_function_steepest
from tverrsnitt import Tverrsnitt

def integrate_cross_section(
    eps_ok: float,
    eps_uk: float,
    height_ec_zero: float,
    height_total: float,
    material: ConcreteMaterial,
    var_height: float = None,
    creep_eff: float = 0,
) -> Tuple[float, float]:
    """Integrere opp betongarealet"""
    # d_alpha_d er avstanden fra trykkresultanten til stedet nøytralaksen
    sum_f = 0
    sum_mom = 0
    height_compression = height_total - height_ec_zero

    iterations = 100
    delta_e = (eps_ok - eps_uk) / iterations
    delta_h = height_compression / iterations
    for i in range(1, iterations):
        height_i = height_ec_zero + delta_h * i
        if var_height is None:
            width_i = get_width2(height_i, 0)
        else:
            width_i = get_width2(height_i, var_height)
            if width_i > 4200: # TODO!
                print("error med width i integrate cross section")

        area_i = width_i * delta_h

        eps_i = eps_uk + delta_e * (i - 0.5)
        sigma_i = material.get_stress(eps_i, creep_eff=creep_eff)
        if sigma_i == 0 and eps_i != 0:
            print(f"eps_i: {eps_i}, sigma_i: {sigma_i}, height_i: {height_i}, delta_h: {delta_h}")
        sigma_i = material.get_stress(eps_i, creep_eff=creep_eff)
        f_i = area_i * sigma_i
        sum_f += f_i
        sum_mom += f_i * (height_i - height_ec_zero - delta_h / 2)
    sum_mom = abs(sum_mom)
    if sum_f == 0:
        print(f"Feil i integrering av tverrsnitt. eps_ok: {eps_ok}, eps_uk: {eps_uk}, sum_f: {sum_f}")
        sys.exit(1)
    d_alpha_d = sum_mom / abs(sum_f)

    return sum_f, d_alpha_d


def evaluate_reinforcement_from_strain(
    d_vector: ndarray,
    a_vector: ndarray,
    height: float,
    eps_ok: float,
    eps_uk: float,
    steel_material: Material,
    concrete_material: ConcreteMaterial,
    is_inside_concrete: bool,
    is_ck_not_cd: bool = True,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Krefter i armering basert på tøyninger. Behandler alle som punkter og ikke areal"""
    d_vec_strekk: ndarray = array([])
    f_vec_strekk: ndarray = array([])
    d_vec_trykk: ndarray = array([])
    f_vec_trykk: ndarray = array([])

    for d, as_ in zip(d_vector, a_vector):
        toyning = eps_ok + (eps_uk - eps_ok) / height * d # Geometrisk tøyning
        spenning = steel_material.get_stress(toyning, is_ck_not_cd=is_ck_not_cd)

        if d == 0 or as_ == 0:
            pass

        elif spenning >= 0:
            # Tension
            d_vec_strekk = np.append(d_vec_strekk, d)
            f_vec_strekk = np.append(f_vec_strekk, spenning * as_)

        else:
            # Compression - using adjusted stress to account for displaced concrete area
            if is_inside_concrete:
                betong_spenning = concrete_material.get_stress(toyning)
            else:
                betong_spenning = 0
            justert_spenning = spenning - betong_spenning

            # Sjekker om det er spenningarmering, må i så fall legge til forspenningskraften
            d_vec_trykk = np.append(d_vec_trykk, d)
            f_vec_trykk = np.append(f_vec_trykk, justert_spenning * as_)

    return d_vec_strekk, f_vec_strekk, d_vec_trykk, f_vec_trykk


def section_integrator(
    eps_ok: float,
    eps_uk: float,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial = None,
    creep: float = 0,
    tendon_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
) -> Tuple[float, float, float, float]:
    """Integrerer opp tverrsnittet"""
    #eps_ok, eps_uk, tverrsnitt, material, rebar_material, rebar_pre_material,
    #    carbon_material, creep_eff, is_ck_not_cd
    # Starter med å finne alpha fra tøyningene. Antar at tøyninger som gir trykk er positive, og strekk negativt.
    # delta_eps: float = (eps_ok - eps_uk) / height
    d_bot = tverrsnitt.get_d_bot()
    d_top = tverrsnitt.get_d_top()
    height = tverrsnitt.get_height()
    height_max = tverrsnitt.get_height_max()
    d_pre_bot = tverrsnitt.get_d_pre_bot()
    d_pre_top = tverrsnitt.get_d_pre_top()
    d_carbon = tverrsnitt.get_d_carbon()
    a_carbon = tverrsnitt.get_a_carbon()
    as_bot = tverrsnitt.get_as_area_bot()
    as_top = tverrsnitt.get_as_area_top()
    a_pre_bot = tverrsnitt.get_a_pre_bot()
    a_pre_top = tverrsnitt.get_a_pre_top()
    
    if len(d_bot) == 0:
        d_bot_0 = 0
    else:
        d_bot_0 = d_bot[0]
    if len(d_pre_bot) == 0:
        d_pre_bot_0 = 0
    else:
        d_pre_bot_0 = d_pre_bot[0]

    d_0 = max(d_bot_0, d_pre_bot_0)
    delta_eps: float = (eps_uk - eps_ok) / height_max
    eps_s = eps_uk - delta_eps * (height - d_0)  # Geometrisk tøyning i sone 0
    # eps_s_d0: float = eps_uk + delta_eps * d_0
    eps_s_d0 = eps_s
    alpha: float = min(max(-eps_ok / (eps_s_d0 - eps_ok), 0), 1)
    """if __debug__ and alpha in (0, 1):
        # Ugyldig verdi, feil i utregning
        print(f"feil i alpha. eps_ok: {eps_ok:.6f}, eps_uk: {eps_uk:.6f}", alpha)"""

    # Ønsker å finne hvilke lag som har strekk og trykk (og størrelse på kreftene)
    # Slakkarmering
    if rebar_material is not None:
        # Starter med å samle d- og areal-vektorer
        d_vector = np.concatenate((d_bot, d_top), axis=0)
        rebar_vector = np.concatenate((as_bot, as_top), axis=0)
        
        # Regner ut kraft, og sorterer etter strekk og trykk
        d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(
            d_vector, rebar_vector, height, eps_ok, eps_uk, rebar_material, material, True, is_ck_not_cd,
        )
    else:
        f_strekk: float = 0
        f_trykk: float = 0
        d_strekk_avg: float = 0
        d_trykk_avg: float = 0
        d_strekk: ndarray = np.array([])
    
    # Summerer kreftene (selv om de er 0)
    sum_f_strekk_armering: float = np.sum(f_strekk)
    sum_f_trykk_armering: float = np.sum(f_trykk)
    if sum_f_strekk_armering == 0:
        d_strekk_rebar: float = 0
    else:
        d_strekk_rebar: float = np.dot(f_strekk, d_strekk) / max(sum_f_strekk_armering, 1)

    if tendon_material is not None:
        if d_pre_top is not None:
            d_vector_tendon: ndarray = np.concatenate((d_pre_bot, d_pre_top), axis=0)
            tendon_area_vector: ndarray = np.concatenate((a_pre_bot, a_pre_top), axis=0)
        else:
            d_vector_tendon: ndarray = d_pre_bot
            tendon_area_vector: ndarray = a_pre_bot

        d_strekk_tendon, f_strekk_tendon_vec, d_trykk_tendon, f_trykk_tendon_vec = (
            evaluate_reinforcement_from_strain(
                d_vector_tendon,
                tendon_area_vector,
                height,
                eps_ok,
                eps_uk,
                steel_material=tendon_material,
                concrete_material=material,
                is_inside_concrete=False,
                is_ck_not_cd=is_ck_not_cd,
            )
            # f_strekk_tendon er i milli-N?
        )
        f_strekk_tendon: float = np.sum(f_strekk_tendon_vec)
        f_trykk_tendon: float = np.sum(f_trykk_tendon_vec)
        if f_strekk_tendon == 0:
            d_strekk_tendon_avg: float = 0
        else:
            d_strekk_tendon_avg: float = (
                np.dot(f_strekk_tendon_vec, d_strekk_tendon) / f_strekk_tendon
            )

        # Må sjekke om trykk er 0 før deling
        if f_trykk_tendon == 0:
            d_trykk_tendon_avg: float = 0
        else:
            d_trykk_tendon_avg: float = (
                np.dot(f_trykk_tendon_vec, d_trykk_tendon) / f_trykk_tendon
            )

    else:
        f_strekk_tendon: float = 0
        f_trykk_tendon: float = 0
        d_strekk_tendon_avg: float = 0
        d_trykk_tendon_avg: float = 0

    if carbon_material is not None:
        d_strekk_karbon, f_strekk_karbon_vec, d_trykk_karbon, f_trykk_karbon_vec = (
            evaluate_reinforcement_from_strain(
                d_carbon,
                a_carbon,
                height,
                eps_ok,
                eps_uk,
                carbon_material,
                material,
                False,
                is_ck_not_cd,
            )
        )
        f_strekk_karbon: float = np.sum(f_strekk_karbon_vec)
        f_trykk_karbon: float = np.sum(f_trykk_karbon_vec)
        m_strekk_karbon: float = np.dot(f_strekk_karbon_vec, d_strekk_karbon)
        m_trykk_karbon: float = np.dot(f_trykk_karbon_vec, d_trykk_karbon)
        if f_strekk_karbon == 0:
            d_strekk_karbon_avg: float = 0
        else:
            d_strekk_karbon_avg: float = (
                m_strekk_karbon / f_strekk_karbon if f_strekk_karbon > 0 else 0
            )

        # Må sjekke om trykk er 0 før deling
        if f_trykk_karbon == 0:
            d_trykk_karbon_avg: float = 0
        else:
            d_trykk_karbon_avg: float = m_trykk_karbon / f_trykk_karbon

    else:
        f_strekk_karbon: float = 0
        f_trykk_karbon: float = 0
        d_strekk_karbon_avg: float = 0
        d_trykk_karbon_avg: float = 0

    # Tyngdepunkt for armering
    #if sum_f_strekk_armering == 0 and f_strekk_tendon == 0:
    #    print("sum_strekkrefter er 0")
    if sum_f_strekk_armering > 0 or f_strekk_tendon > 0:
        d_strekk_avg: float = (
            d_strekk_rebar * sum_f_strekk_armering
            + d_strekk_tendon_avg * f_strekk_tendon
            + d_strekk_karbon_avg * f_strekk_karbon
        ) / (sum_f_strekk_armering + f_strekk_tendon + f_strekk_karbon)
    else:
        d_strekk_avg: float = 0

    # Summerer strekkbidragene
    sum_strekk = sum_f_strekk_armering + f_strekk_tendon + f_strekk_karbon

    alpha_d: float = alpha * d_0
    eps_c_uk = 0.0  # bøyning, en del vil alltid være i strekk så setter denne 0 for integralet sin del

    height_uk = height - alpha_d
    f_bet, d_alpha_d = integrate_cross_section(
        eps_ok, eps_c_uk, height_uk, height, material, var_height=1180, creep_eff=creep,
    )  # , var_height)
    d_bet = alpha_d - d_alpha_d

    # Regner ut bidraget fra armering. Trykkmoment regnes om overkant
    sum_trykkmoment: float = (
        np.dot(f_trykk, d_trykk)
        + f_trykk_tendon * d_trykk_tendon_avg
        + f_trykk_karbon * d_trykk_karbon_avg
        + f_bet * d_bet
    )
    sum_trykk: float = sum_f_trykk_armering + f_trykk_tendon + f_trykk_karbon + f_bet

    # d_trykk_avg er målt fra OK
    d_trykk_avg: float = abs(sum_trykkmoment / sum_trykk)

    z = d_strekk_avg - d_trykk_avg

    return alpha, sum_trykk, sum_strekk, z


def objective_function_eps_s(
    eps_s,
    eps_c,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    # eps_uk = eps_s / (d0 * (1 - alpha)) * (height - d0 * alpha)
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_b, f_s, z = section_integrator(
        eps_c,
        eps_s,
        tverrsnitt,
        material,
        rebar_material,
        creep_eff,
        tendon_material=rebar_pre_material,
        carbon_material=carbon_material,
        is_ck_not_cd=is_ck_not_cd,
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = abs(f_b * z)
    if abs(abs(mom_b) / max(abs(mom_s), 1e-6) - 1) < 0.001:
        # print("konv")
        # print("f_s: ", f_s, "f_b:", f_b, "alpha: ", alpha)
        pass

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_s / max(mom_b, 1e-6) - 1.0, alpha, mom_s, mom_b, z


def objective_function_eps_c(
    eps_s,
    eps_c,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    creep_eff: float = 0,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    eps_ok, eps_uk = eps_c_and_eps_s_to_eps_ok_uk(eps_c, eps_s, tverrsnitt.get_height_max(), tverrsnitt.get_d0_bot())
    alpha, f_b, f_s, z = section_integrator(
        eps_ok,
        eps_uk,
        tverrsnitt,
        material,
        rebar_material,
        creep_eff,
        tendon_material=rebar_pre_material,
        carbon_material=carbon_material,
        is_ck_not_cd=is_ck_not_cd,
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = abs(f_b * z)

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_b / max(mom_s, 1e-6) - 1.0, alpha, mom_s, mom_b, z


def newton_optimize_eps_s(
    eps_c,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    initial_guess,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
):
    """
    Metode som gjør iterasjonen for eps_s via Newton-Raphson.
    Inneholder egne sjekk for eps_s-optimering og er derfor skilt fra eps_c.
    """
    # Initialiserer
    max_iterations = 30
    tolerance = 1e-3
    eps_s = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0

    # Finner eps_s_u
    if rebar_material is not None:
        eps_s_rebar = rebar_material.get_eps_s_u()
    else:
        eps_s_rebar = 99.
    if rebar_pre_material is not None:
        assert isinstance(rebar_pre_material, Tendon)
        eps_s_pre = rebar_pre_material.get_max_external_strain()
    else:
        eps_s_pre = 99.
    if carbon_material is not None:
        assert isinstance(carbon_material, CarbonFiber)
        eps_s_cf = carbon_material.get_eps_s_u()
    else:
        eps_s_cf = 99.

    # Velger minste bruddtøyning som blir dimensjonerende
    eps_s_u = min(eps_s_rebar, eps_s_pre, eps_s_cf)

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_s, mom_b, z = objective_function_eps_s(
            eps_s,
            eps_c,
            tverrsnitt,
            material,
            rebar_material,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _, _ = objective_function_eps_s(
            eps_s + h,
            eps_c,
            tverrsnitt,
            material,
            rebar_material,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )

        f_prime = (f_value2 - f_value) / h

        # Oppdaterer eps_s basert på Newton-Raphson-metoden
        eps_s -= step_size * f_value / f_prime

        # Sjekk om tøyningen er altfor stor
        if eps_s > eps_s_u:
            if mom_s < mom_b:
                return -1.0, -1.0, -1.0, -1.0
            eps_s = eps_s_u - 0.004
            step_size *= 0.5  # Kan kanskje på sikt fjerne step_size.

        # Sjekk mot konvergenskriteriet
        if abs_f_value < tolerance:
            return eps_s, alpha, mom_b, z
        if eps_s < 0.0 or math.isnan(eps_s):
            eps_s = 0.0001
            step_size *= 0.75

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0


def newton_optimize_eps_c(
    eps_s,
    tverrsnitt: Tverrsnitt,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    initial_guess,
    eps_cu,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    is_ck_not_cd: bool = True,
):
    """
    Metode som gjør iterasjonen for eps_c via Newton-Raphson.
    Inneholder egne sjekk for eps_c-optimering og er derfor skilt fra eps_s.
    """
    # Initialiserer
    max_iterations = 20
    tolerance = 1e-3
    eps_c = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0
    fortegn_tracker = [2, 2, 2]

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_s, mom_b, z = objective_function_eps_c(
            eps_s,
            eps_c,
            tverrsnitt,
            material,
            rebar_material,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )
        abs_f_value = abs(f_value)
        fortegn_tracker[iterations % 3] = np.sign(f_value)

        f_value2, _, _, _ , _= objective_function_eps_c(
            eps_s,
            eps_c + h,
            tverrsnitt,
            material,
            rebar_material,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )

        f_prime = (f_value2 - f_value) / h

        # Justerer verdi i henhold til Newton-Raphson-iterasjonen
        if abs_f_value > 1e-6:
            if iterations == 17:
                step_size *= 0.25
            eps_c -= step_size * f_value / f_prime

            # Sjekk om tøyningen er altfor stor
            if eps_c < eps_cu:
                eps_c = eps_cu
                step_size *= 0.5  # Kan kanskje på sikt fjerne step_size.
            elif eps_c > 0:
                step_size *= 0.85
                eps_c = -0.0001
            elif fortegn_tracker in (
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
            ):
                step_size *= 0.95
                fortegn_tracker = [2, 2, 2]
                max_iterations = 150
            elif iterations == 100:
                step_size *= 0.5

        # Sjekk mot konvergenskriteriet
        if abs_f_value < tolerance:
            return eps_c, alpha, mom_b, z

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0, -1.0


def integration_iterator_ultimate(
    tverrsnitt: Tverrsnitt,
    concrete_material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
    creep_eff: float = 0,
) -> float:
    """For ULS"""
    # Moment_zeroState er momentet som ligger i momentmaks når karbonfiber limes og monteres.
    # Må ha en initiell testverdi
    initial_guess = 0.015
    is_ck_not_cd = False  # Bruddgrensetilstand
    eps_cu = concrete_material.get_eps_cu()
    eps_c = eps_cu
    

    # Kalkulasjon starter
    # Må finne den verdien/kombinasjonen av eps_cu og eps_s som gir indre likevekt i tverrsnittet.
    # Starter med å anta eps_c = eps_cu og ser om ulike verdier av eps_s kan gi likevekt.

    sum_as_bot = np.sum(tverrsnitt.get_as_area_bot())
    sum_as_top = np.sum(tverrsnitt.get_as_area_top())

    alpha, mom_b, z = 0, 0, 0
    if sum_as_bot > sum_as_top or rebar_pre_material is not None or carbon_material is not None:
        eps_s, alpha, mom_b, z = newton_optimize_eps_s(
            eps_c,
            tverrsnitt,
            concrete_material,
            rebar_material,
            initial_guess,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )
    else:
        eps_s = -1.0

    # Sjekker om første iterasjon var vellykket
    if eps_s == -1.0:
        print("Justerer eps_c")
        # Betongtøyningen kan ikke nå eps_cu. Setter en armeringstøyning og finner
        # betongtøyning som gir likevekt i tverrsnittet (eps_c < eps_cu).
        if rebar_material is not None:
            eps_s_rebar = rebar_material.get_eps_s_u()
        else:
            eps_s_rebar = 99.
        if rebar_pre_material is not None:
            assert isinstance(rebar_pre_material, Tendon)
            eps_s_pre = rebar_pre_material.get_max_external_strain()
        else:
            eps_s_pre = 99.
        if carbon_material is not None:
            assert isinstance(carbon_material, CarbonFiber)
            eps_s_cf = carbon_material.get_eps_s_u()
        else:
            eps_s_cf = 99.

        # Velger minste bruddtøyning som blir dimensjonerende
        eps_s = min(eps_s_rebar, eps_s_pre, eps_s_cf)

        initial_guess = -0.002  # -0.000297 #-0.00117
        eps_c, alpha, mom_b, z = newton_optimize_eps_c(
            eps_s,
            tverrsnitt,
            concrete_material,
            rebar_material,
            initial_guess,
            eps_cu,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            carbon_material=carbon_material,
            is_ck_not_cd=is_ck_not_cd,
        )

    return (alpha, mom_b, eps_s, eps_c, z)

def innerstate_beam(eps_ok: float, eps_uk: float, tverrsnitt: Tverrsnitt, moment: float,
                    material: ConcreteMaterial = None, rebar_material: RebarMaterial = None, rebar_pre_material: RebarMaterial = None,
                    carbon_material: CarbonMaterial = None, creep_eff: float = 0,
                    is_ck_not_cd: bool = True) -> Tuple[float, float]:
    if eps_ok == 0. and eps_uk == 0.:
        # Hvis ingen tøyninger, returner 0 krefter
        return 0.0, -moment
    _alpha, f_trykk, f_strekk, z_arm = section_integrator(
        eps_ok, eps_uk, tverrsnitt, material, rebar_material, creep=creep_eff,
        tendon_material=rebar_pre_material, carbon_material=carbon_material, is_ck_not_cd=is_ck_not_cd
    )
    
    # Gjør om z til m for å få mpoment i Nm
    mom = -f_trykk * z_arm / 2 / 1000 + f_strekk * z_arm / 2 / 1000
    f_sum = f_strekk + f_trykk
    m_sum = mom - moment
    return f_sum, m_sum


def find_equilibrium_strains(moment: float, material: ConcreteMaterial,
                             tverrsnitt: Tverrsnitt,
                             rebar_material: RebarMaterial = None,
                             rebar_pre_material: RebarMaterial = None,
                             carbon_material: CarbonMaterial = None,
                             creep_eff: float = 0,
                             eps_ok: float = -1e-6, eps_uk: float = 1e-6,
                             is_ck_not_cd: bool = True,
                             ) -> Tuple[float, float, float, float]:
    """Finner likevektstøyninger for et gitt moment"""
    tolerance, regularization, prev_norm, delta = 0.001, 1e-9, 1e12, 1e-8
    max_iterations, incr = 150, 0.0009
    delta_max = 1e-5
    eps_cu_eff: float = material.get_eps_cu() * (1 + creep_eff) if is_ck_not_cd else material.get_eps_cu()
    if rebar_material is not None:
        eps_s_u: float = rebar_material.get_eps_s_u()
    elif rebar_pre_material is not None:
        eps_s_u: float = rebar_pre_material.get_eps_s_u()
    else:
        eps_s_u: float = 0.02
        print("Warning: No rebar material provided, using default eps_s_u = 0.02")

    base: float = 1
    
    # Starter å iterere
    for i in range(max_iterations):
        f_internal, m_internal = innerstate_beam(eps_ok, eps_uk, tverrsnitt,
                moment, material, rebar_material, rebar_pre_material, carbon_material=carbon_material,
                creep_eff=creep_eff, is_ck_not_cd=is_ck_not_cd)
        current_norm = np.sqrt(f_internal ** 2 + m_internal ** 2)
        
        # Check for NaN in f_internal
        if np.isnan(f_internal):
            print("NaN detected in f_internal, aborting iteration.")
            print(f"Error. eps_ok:{eps_ok:.7f}, eps_uk:{eps_uk:.7f}. Iteration: {i}")
            sys.exit()
        
        if abs(f_internal) < tolerance and m_internal < tolerance and i > 0:
            # Konvergens oppnådd
            return eps_ok, eps_uk, f_internal, m_internal
        if i % 50 == 0 and i > 0:
            # Reduserer steglengden
            incr = max(incr * 0.5, 0.00005)
            #print(f"Reducing step size to {incr} at iteration {i}")
        
        # implementer metode for å flytte oss i løsningsrommet for å finne bedre gradienter
        if i % 23 == 0 and i > 0:
            if i > 70:
                steps, search_step = 2, 0.00005
            else:
                steps, search_step = 7, 0.000005
            eps_ok, eps_uk = help_function_grid(eps_ok, eps_uk, moment, tverrsnitt, material,
                                                rebar_material, rebar_pre_material, carbon_material,
                                                is_ck_not_cd, creep_eff, steps, search_step)
            # Bør erstattes på sikt
            if eps_uk <= 0.:
                eps_uk = 1.1e-8 + i * 1e-9
            if eps_ok >= 0.:
                eps_ok = -1.1e-8 - i * 1e-9
            continue
        if i % 1400 == 0 and i > 0: # endre til i % 14 på sikt
            steps: float = 7 if i > 30 else 15
            eps_ok, eps_uk = help_function_steepest(eps_ok, eps_uk, moment,
                    tverrsnitt, material, rebar_material, rebar_pre_material,
                    carbon_material=carbon_material, is_ck_not_cd=is_ck_not_cd,
                    creep_eff=creep_eff, steps=steps)
            continue
        
        f1, m1 = innerstate_beam(eps_ok + delta, eps_uk, tverrsnitt,
            moment, material, rebar_material, rebar_pre_material, carbon_material=carbon_material,
            creep_eff=creep_eff, is_ck_not_cd=is_ck_not_cd)
        f2, m2 = innerstate_beam(eps_ok, eps_uk + delta, tverrsnitt,
            moment, material, rebar_material, rebar_pre_material, carbon_material=carbon_material,
            creep_eff=creep_eff, is_ck_not_cd=is_ck_not_cd)
        
        # Construct the Jacobian matrix (2x2)
        j = np.array([
            [(f1 - f_internal) / delta, (f2 - f_internal) / delta],
            [(m1 - m_internal) / delta, (m2 - m_internal) / delta]
        ]) + np.eye(2) * regularization

        # Residual vector
        r = np.array([f_internal, m_internal])

        # Solve for delta_epsilon: j * delta_epsilon = -r
        try:
            delta_epsilon = np.linalg.solve(j, -r)
        except np.linalg.LinAlgError:
            print("Matrix is singular")
            break

        # Adaptive step size adjustment based on residual norm improvement
        if current_norm < prev_norm:
            base = min(base * 1.1, 1.0)
        else:
            base *= 0.85 # Reduseres ved ingen forbedring
    
        delta_top = min(abs(delta_epsilon[0] * base), delta_max) * np.sign(delta_epsilon[0])
        delta_bottom = min(abs(delta_epsilon[1] * base), delta_max) * np.sign(delta_epsilon[1]) # obs obs er min max feil?

        # Update strains
        # Clamp the strains to avoid overshooting boundaries
        #eps_ok = max(min(eps_ok + delta_top, eps_s_u), eps_cu_eff)
        #eps_uk = max(min(eps_uk + delta_bottom, eps_s_u), eps_cu_eff)
        eps_ok = max(eps_ok + delta_top, eps_cu_eff)
        eps_uk = min(eps_uk + delta_bottom, eps_s_u)
        
        # Hvis 0 tøyning
        if eps_uk == 0 and eps_ok == 0:
            eps_uk = 0.00070 + i * 0.00001
            
        # Bør erstattes på sikt
        if eps_uk <= 0.:
            eps_uk = 1.1e-8 + i * 1e-9
            base *= 0.9
        if eps_ok >= 0.:
            eps_ok = -1.1e-8 - i * 1e-9
            base *= 0.9
        
        # Oppdaterer prev_norm for neste iterasjon
        prev_norm = current_norm
        
        #if i == 90:
            #print("Over 90 iterasjoner")
    print("Max iterations reached without convergence")
        
    return 0, 0, 0, 0


def get_width2(_a: float, _b: float) -> float:
    """Dummy, men fungerer for konstant bredde"""
    return 300.0


def get_width(height_i: float, var: float) -> float:
    """Lager en funksjon som beskriver bredden for enhver høyde"""
    if height_i < 80:
        return 320
    if height_i < 220:
        return 320 - 220 * (height_i - 80) / 140
    if height_i < 220 + var:
        return 100
    if height_i < 220 + var + 50:
        rel_height = height_i - (220 + var)
        return 100 + 320 * rel_height / 50
    return 420

