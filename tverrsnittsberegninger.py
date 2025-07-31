"""Tverrsnittsberegninger for karbonfiberforsterket spennarmert betongtverrsnitt"""

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


def integrate_cross_section(
    eps_ok: float,
    eps_uk: float,
    height_ec_zero: float,
    height_total: float,
    material: ConcreteMaterial,
    var_height: float = None,
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
            width_i = get_width(height_i, 0)
        else:
            width_i = get_width(height_i, var_height)
            if width_i > 420:
                print("error med width i integrate cross section")

        area_i = width_i * delta_h

        eps_i = eps_uk + delta_e * (i - 0.5)
        sigma_i = material.get_stress(eps_i)
        f_i = area_i * sigma_i
        sum_f += f_i
        sum_mom += f_i * (height_i - height_ec_zero - delta_h / 2)
    sum_mom = abs(sum_mom)
    d_alpha_d = sum_mom / abs(sum_f)

    return sum_f, d_alpha_d


def evaluate_reinforcement_from_strain(
    d_vector: ndarray,
    a_vector: ndarray,
    d_0: float,
    eps_c: float,
    eps_s: float,
    steel_material: Material,
    concrete_material: ConcreteMaterial,
    is_inside_concrete: bool,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Krefter i armering basert på tøyninger. Behandler alle som punkter og ikke areal"""
    d_vec_strekk: ndarray = array([])
    f_vec_strekk: ndarray = array([])
    d_vec_trykk: ndarray = array([])
    f_vec_trykk: ndarray = array([])

    for d, as_ in zip(d_vector, a_vector):
        toyning = eps_c + (eps_s - eps_c) / d_0 * d
        spenning = steel_material.get_stress(toyning)

        if d == 0 or as_ == 0:
            pass

        elif spenning >= 0:
            # Tension
            d_vec_strekk = np.append(d_vec_strekk, d)
            f_vec_strekk = np.append(f_vec_strekk, spenning * as_)

        else:
            # Compression - using adjusted stress to account for displaced concrete area
            if is_inside_concrete:
                concrete_spenning = concrete_material.get_stress(toyning)
            else:
                concrete_spenning = 0
            justert_spenning = spenning - concrete_spenning

            # Sjekker om det er spenningarmering, må i så fall legge til forspenningskraften
            d_vec_trykk = np.append(d_vec_trykk, d)
            f_vec_trykk = np.append(f_vec_trykk, justert_spenning * as_)

    return d_vec_strekk, f_vec_strekk, d_vec_trykk, f_vec_trykk


def section_integrator(
    eps_ok: float,
    eps_s: float,
    height: float,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    as_bot: ndarray,
    as_top: ndarray,
    d_bot: ndarray,
    d_top: ndarray,
    creep: float,
    a_pre_bot: ndarray = None,
    a_pre_top: ndarray = None,
    d_pre_bot: ndarray = None,
    d_pre_top: ndarray = None,
    tendon_material: RebarMaterial = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    moment_zero_state: float = 0
) -> Tuple[float, float, float, float]:
    """Integrerer opp tverrsnittet"""
    # Starter med å finne alpha fra tøyningene. Antar at tøyninger som gir trykk er positive, og strekk negativt.
    # delta_eps: float = (eps_ok - eps_uk) / height
    d_0 = max(d_bot[0], d_pre_bot[0])
    delta_eps: float = (eps_s - eps_ok) / d_0
    eps_uk: float = eps_s + delta_eps * (height - d_0)
    # eps_s_d0: float = eps_uk + delta_eps * d_0
    eps_s_d0 = eps_s
    alpha: float = min(max(eps_ok / (eps_ok - eps_s_d0), 0), 1)
    if alpha in (0, 1):
        print("feil i alpha", alpha)

    d_vector = np.concatenate((d_bot, d_top), axis=0)
    rebar_vector = np.concatenate((as_bot, as_top), axis=0)

    # Ønsker å finne hvilke lag som har strekk og trykk (og størrelse på kreftene)
    d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(
        d_vector, rebar_vector, d_0, eps_ok, eps_s_d0, rebar_material, material, True
    )

    sum_f_strekk_armering = np.sum(f_strekk)
    sum_f_trykk_armering = np.sum(f_trykk)
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
                d_0,
                eps_ok,
                eps_s,
                tendon_material,
                material,
                False,
            )
            # TODO! antar at eps_s gjelder for både armering og vaiere
        )
        f_strekk_tendon: float = np.sum(f_strekk_tendon_vec)
        f_trykk_tendon: float = np.sum(f_trykk_tendon_vec)
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

    if a_carbon is not None:
        d_strekk_karbon, f_strekk_karbon_vec, d_trykk_karbon, f_trykk_karbon_vec = (
            evaluate_reinforcement_from_strain(
                d_carbon,
                a_carbon,
                d_0,
                eps_ok,
                eps_s_d0,
                carbon_material,
                material,
                False,
            )
        )
        f_strekk_karbon: float = np.sum(f_strekk_karbon_vec)
        f_trykk_karbon: float = np.sum(f_trykk_karbon_vec)
        m_strekk_karbon: float = np.dot(f_strekk_karbon_vec, d_strekk_karbon)
        m_trykk_karbon: float = np.dot(f_trykk_karbon_vec, d_trykk_karbon)
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
    if sum_f_strekk_armering == 0 and f_strekk_tendon == 0:
        print("sum_f_ er 0")
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
        eps_ok, eps_c_uk, height_uk, height, material, var_height=1180
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
    height,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    as_bot,
    d_bot,
    a_pre_bot,
    d_pre_bot,
    as_top,
    d_top,
    a_pre_top,
    d_pre_top,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    moment_zero_state: float = 0,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    # eps_uk = eps_s / (d0 * (1 - alpha)) * (height - d0 * alpha)
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_b, f_s, z = section_integrator(
        eps_c,
        eps_s,
        height,
        material,
        rebar_material,
        as_bot,
        as_top,
        d_bot,
        d_top,
        creep_eff,
        a_pre_bot=a_pre_bot,
        a_pre_top=a_pre_top,
        d_pre_bot=d_pre_bot,
        d_pre_top=d_pre_top,
        tendon_material=rebar_pre_material,
        a_carbon=a_carbon,
        d_carbon=d_carbon,
        carbon_material=carbon_material,
        moment_zero_state=moment_zero_state
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
    height: ndarray,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    as_bot,
    d_bot,
    a_pre_bot,
    d_pre_bot,
    as_top,
    d_top,
    a_pre_top,
    d_pre_top,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    moment_zero_state: float = 0
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_b, f_s, z = section_integrator(
        eps_c,
        eps_s,
        height,
        material,
        rebar_material,
        as_bot,
        as_top,
        d_bot,
        d_top,
        creep_eff,
        a_pre_bot=a_pre_bot,
        a_pre_top=a_pre_top,
        d_pre_bot=d_pre_bot,
        d_pre_top=d_pre_top,
        tendon_material=rebar_pre_material,
        a_carbon=a_carbon,
        d_carbon=d_carbon,
        carbon_material=carbon_material,
        moment_zero_state=moment_zero_state,
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = abs(f_b * z)

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_b / max(mom_s, 1e-6) - 1.0, alpha, mom_b, z


def newton_optimize_eps_s(
    eps_c,
    height: ndarray,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
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
    rebar_pre_material: RebarMaterial = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    moment_zero_state: float = 0
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

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_s, mom_b, z = objective_function_eps_s(
            eps_s,
            eps_c,
            height,
            material,
            rebar_material,
            as_bot,
            d_bot,
            a_pre_bot,
            d_pre_bot,
            as_top,
            d_top,
            a_pre_top,
            d_pre_top,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _, _ = objective_function_eps_s(
            eps_s + h,
            eps_c,
            height,
            material,
            rebar_material,
            as_bot,
            d_bot,
            a_pre_bot,
            d_pre_bot,
            as_top,
            d_top,
            a_pre_top,
            d_pre_top,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
            moment_zero_state=moment_zero_state,
        )

        f_prime = (f_value2 - f_value) / h

        # Oppdaterer eps_s basert på Newton-Raphson-metoden
        eps_s -= step_size * f_value / f_prime

        # Sjekk om tøyningen er altfor stor
        if eps_s > 0.02:
            if mom_s < mom_b:
                return -1.0, -1.0, -1.0, -1.0
            eps_s = 0.016
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
    height: ndarray,
    material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    as_bot,
    as_top,
    d_bot,
    d_top,
    a_pre_bot,
    a_pre_top,
    d_pre_bot,
    d_pre_top,
    initial_guess,
    eps_cu,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    moment_zero_state: float = 0
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
        f_value, alpha, mom_b, z = objective_function_eps_c(
            eps_s,
            eps_c,
            height,
            material,
            rebar_material,
            as_bot,
            d_bot,
            a_pre_bot,
            d_pre_bot,
            as_top,
            d_top,
            a_pre_top,
            d_pre_top,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
            moment_zero_state=moment_zero_state,
        )
        abs_f_value = abs(f_value)
        fortegn_tracker[iterations % 3] = np.sign(f_value)

        f_value2, _, _, _ = objective_function_eps_c(
            eps_s,
            eps_c + h,
            height,
            material,
            rebar_material,
            as_bot,
            d_bot,
            a_pre_bot,
            d_pre_bot,
            as_top,
            d_top,
            a_pre_top,
            d_pre_top,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
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
            print("Iteration: ", iterations)
            return eps_c, alpha, mom_b, z

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0


def integration_iterator_ultimate(
    height: ndarray,
    as_area_bot: ndarray,
    as_area_top: ndarray,
    d_bot: ndarray,
    d_top: ndarray,
    concrete_material: ConcreteMaterial,
    rebar_material: RebarMaterial,
    rebar_pre_material: RebarMaterial = None,
    a_pre_bot: ndarray = None,
    a_pre_top: ndarray = None,
    d_pre_bot: ndarray = None,
    d_pre_top: ndarray = None,
    a_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    creep_eff: float = 0,
    moment_zero_state: float = 0
) -> float:
    """For ULS"""
    # Moment_zeroState er momentet som ligger i momentmaks når karbonfiber limes og monteres.
    # Må ha en initiell testverdi
    initial_guess = 0.015
    eps_cu = concrete_material.get_eps_cu()
    eps_c = eps_cu

    # Kalkulasjon starter
    # Må finne den verdien/kombinasjonen av eps_cu og eps_s som gir indre likevekt i tverrsnittet.
    # Starter med å anta eps_c = eps_cu og ser om ulike verdier av eps_s kan gi likevekt.

    sum_as_bot = np.sum(as_area_bot)
    sum_as_top = np.sum(as_area_top)

    alpha, mom_b, z = 0, 0, 0
    if sum_as_bot > sum_as_top or rebar_pre_material is not None:
        eps_s, alpha, mom_b, z = newton_optimize_eps_s(
            eps_c,
            height,
            concrete_material,
            rebar_material,
            as_area_bot,
            as_area_top,
            d_bot,
            d_top,
            a_pre_bot,
            a_pre_top,
            d_pre_bot,
            d_pre_top,
            initial_guess,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
            moment_zero_state=moment_zero_state,
        )
    else:
        eps_s = -1.0

    # Sjekker om første iterasjon var vellykket
    if eps_s == -1.0:
        print("Justerer eps_c")
        # Betongtøyningen kan ikke nå eps_cu. Setter en armeringstøyning og finner
        # betongtøyning som gir likevekt i tverrsnittet (eps_c < eps_cu).
        if isinstance(rebar_pre_material, Tendon):
            assert isinstance(rebar_pre_material, Tendon)
            eps_s = rebar_pre_material.get_max_external_strain()
        else:
            eps_s = 0.02
        initial_guess = -0.002  # -0.000297 #-0.00117
        eps_c, alpha, mom_b, z = newton_optimize_eps_c(
            eps_s,
            height,
            concrete_material,
            rebar_material,
            as_area_bot,
            as_area_top,
            d_bot,
            d_top,
            a_pre_bot,
            a_pre_top,
            d_pre_bot,
            d_pre_top,
            initial_guess,
            eps_cu,
            creep_eff,
            rebar_pre_material=rebar_pre_material,
            a_carbon=a_carbon,
            d_carbon=d_carbon,
            carbon_material=carbon_material,
            moment_zero_state=moment_zero_state,
        )

    return (alpha, mom_b, eps_s, eps_c, z)


def get_width2(_a: float, _b: float) -> float:
    """Dummy, men fungerer for konstant bredde"""
    return 200.0


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


if __name__ == "__main__":
    betong_b45: ConcreteMaterial = ConcreteMaterial(45, material_model="Parabola")
    armering: RebarMaterial = RebarB400NC()
    # armering: RebarMaterial = RebarB500NC()
    spennarmering: RebarMaterial = Tendon()
    spennarmering.prestressd_to(20)
    antall_vector_ok = np.array([2])
    antall_vector_uk = np.array([4, 6, 4, 2])
    area_vector_ok = spennarmering.get_area(antall_vector_ok)
    area_vector_uk = spennarmering.get_area(antall_vector_uk)

    karbonfiber: CarbonMaterial = CarbonFiber()
    carbon_vector: ndarray = np.array([50 * 1.4 * 2])
    # carbon_vector = None  #  eps_s: 0.00912011152199339
    d_carbon: ndarray = np.array([1560])

    height = 1600
    as_area_bot = np.array([0])
    as_area_top = np.array([(2 * 36) * 3.14])
    d_bot = np.array([0])
    d_top = np.array([40])
    d_pre_bot = np.array([1560, 1520, 1480, 1440])
    d_pre_top = np.array([40])

    alpha, mom, eps_s, eps_c, z = integration_iterator_ultimate(
        height,
        as_area_bot,
        as_area_top,
        d_bot,
        d_top,
        betong_b45,
        armering,
        rebar_pre_material=spennarmering,
        a_pre_bot=area_vector_uk,
        d_pre_bot=d_pre_bot,
        # a_pre_top=area_vector_ok,
        # d_pre_top=d_pre_top,
        d_carbon=d_carbon,
        a_carbon=carbon_vector,
        carbon_material=karbonfiber,
        moment_zero_state=10.,
    )
    print("alpha:", alpha, "mom:", mom / 1e6, "eps_c:", eps_c, "eps_s:", eps_s)


def testing_strain_part():
    """Brukt for å teste evaluate reinforcement blant annet"""
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    armering: RebarMaterial = RebarB500NC()
    spennarmering: RebarMaterial = Tendon()
    spennarmering.prestressd_to(115)

    # height = np.array(300)
    height = 300
    as_area_bot = np.array([228 * 3.14, 64 * 3.14])
    as_area_top = np.array([64 * 3.14])
    d_bot = np.array([250, 200])
    d_top = np.array([50])
    alpha, mom, eps_s, eps_c, z = integration_iterator_ultimate(
        height, as_area_bot, as_area_top, d_bot, d_top, betong_b35, armering
    )
    # print("alpha:", alpha, ". eps_c:", eps_c, "eps_s:", eps_s)
    # eps_c_uk: float = 0.0026340921430255495

    d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(
        np.array([250, 200, 50]),
        np.array([228 * 3.14, 64 * 3.14, 64 * 3.14]),
        d_bot[0],
        -0.0035,
        eps_s,
        armering,
        betong_b35,
        True,
    )

    alpha_d = alpha * d_bot[0]
    f_bet, d_bet = integrate_cross_section(
        eps_c, 0, height - alpha_d, height, betong_b35
    )  # , var_height)

    alpha, sum_trykk, sum_strekk_arm, z = section_integrator(
        -0.0035,
        eps_s,
        height,
        betong_b35,
        armering,
        as_area_bot,
        as_area_top,
        d_bot,
        d_top,
        0,
    )
    print(
        f"alpha: {alpha}. Sum_trykk: {sum_trykk}. Sum_stress_arm: {sum_strekk_arm}. z: {z}"
    )

    # print("sum trykk:", f_bet - f_trykk.sum())
    antall_vector = np.array([2, 2])
    area_vector = spennarmering.get_area(antall_vector)
    d_strekk_f, f_strekk_f, d_trykk_f, f_trykk_f = evaluate_reinforcement_from_strain(
        np.array([250, 200]),
        area_vector,
        250,
        betong_b35.get_eps_cu(),
        0.002,
        spennarmering,
        betong_b35,
        True,
    )
    print(f"f_strekk_f: {f_strekk_f}. f_trykk_f: {f_trykk_f}")
