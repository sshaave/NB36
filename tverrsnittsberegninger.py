"""Tverrsnittsberegninger for karbonfiberforsterket spennarmert betongtverrsnitt"""

import math
from typing import Tuple
import numpy as np
from numpy import array, ndarray
from materialmodeller import (
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    Material,
    RebarB500NC,
    Tendon,
)


def integrate_cross_section(
    e_ok: float,
    e_uk: float,
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
    delta_e = (e_ok - e_uk) / iterations
    delta_h = height_compression / iterations
    for i in range(1, iterations):
        height_i = height_ec_zero + delta_h * i
        if var_height is None:
            width_i = get_width(height_i, 0)
        else:
            width_i = get_width(height_i, var_height)
        area_i = width_i * delta_h

        eps_i = e_uk + delta_e * (i - 0.5)
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
    e_c: float,
    e_s: float,
    creep_eff: float,
    steel_material: Material,
    concrete_material: ConcreteMaterial,
    is_inside_concrete: bool,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Krefter i armering basert på tøyninger. Behandler alle som punkter og ikke areal"""
    f_yd = steel_material.get_f_yd()
    e_s_flyt = steel_material.get_eps_s_y()
    emod_s = steel_material.get_e_s_rebar()  # N/mm2

    d_vec_strekk: ndarray = array([])
    f_vec_strekk: ndarray = array([])
    d_vec_trykk: ndarray = array([])
    f_vec_trykk: ndarray = array([])

    if isinstance(steel_material, Tendon):
        assert isinstance(steel_material, Tendon)
        antall_vector = steel_material.get_antall_vec(a_vector)
        f_p = steel_material.get_prestress()  # forspenningskraft
    else:
        antall_vector = np.zeros(a_vector.shape)
        f_p = 0

    for d, as_, antall in zip(d_vector, a_vector, antall_vector):
        toyning = e_c + (e_s - e_c) / d_0 * d
        if toyning >= 0:
            spenning = f_yd if toyning >= e_s_flyt else emod_s * toyning
        else:
            spenning = -f_yd if toyning <= -e_s_flyt else emod_s * toyning

        if toyning >= 0:
            # Tension
            d_vec_strekk = np.append(d_vec_strekk, d)

            # Sjekker om det er spennarmering, og må legge til spennkraft
            if isinstance(steel_material, Tendon):
                assert isinstance(steel_material, Tendon)
                f_vec_strekk = np.append(f_vec_strekk, spenning * as_ + antall * f_p)
            else:
                f_vec_strekk = np.append(f_vec_strekk, spenning * as_)

        else:
            # Compression - using adjusted stress to account for displaced concrete area
            if is_inside_concrete:
                concrete_spenning = concrete_material.get_stress(-toyning)
            else:
                concrete_spenning = 0
            justert_spenning = spenning + concrete_spenning

            # Sjekker om det er spenningarmering, må i så fall legge til forspenningskraften
            d_vec_trykk = np.append(d_vec_trykk, d)
            if isinstance(steel_material, Tendon):
                assert isinstance(steel_material, Tendon)
                f_vec_trykk = np.append(
                    f_vec_trykk, justert_spenning * as_ + antall * f_p
                )
            else:
                f_vec_trykk = np.append(f_vec_trykk, justert_spenning * as_)

    return d_vec_strekk, f_vec_strekk, d_vec_trykk, f_vec_trykk


def section_integrator(
    e_ok: float,
    e_s: float,
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
    d_karbon: ndarray = None,
    carbon_vector: ndarray = None,
    carbon_material: CarbonMaterial = None,
) -> Tuple[float, float, float, float]:
    """Integrerer opp tverrsnittet"""
    # Starter med å finne alpha fra tøyningene. Antar at tøyninger som gir trykk er positive, og strekk negativt.
    # delta_eps: float = (e_ok - e_uk) / height
    d_0 = d_bot[0]
    delta_eps: float = (e_s - e_ok) / d_0
    e_uk: float = e_s + delta_eps * (height - d_0)
    # e_s_d0: float = e_uk + delta_eps * d_0
    e_s_d0 = e_s
    alpha: float = min(max(e_ok / (e_ok - e_s_d0), 0), 1)
    if alpha in (0, 1):
        print("feil i alpha", alpha)

    d_vector = np.concatenate((d_bot, d_top), axis=0)
    rebar_vector = np.concatenate((as_bot, as_top), axis=0)

    # Ønsker å finne hvilke lag som har strekk og trykk (og størrelse på kreftene)
    d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(
        d_vector, rebar_vector, d_0, e_ok, e_s_d0, creep, rebar_material, material, True
    )
    sum_f_strekk_armering = np.sum(f_strekk)
    sum_f_trykk_armering = np.sum(f_trykk)

    if tendon_material is not None:
        if d_pre_top is not None:
            d_vector_tendon: ndarray = np.concatenate((d_pre_bot, d_pre_top), axis=0)
            tendon_area_vector: ndarray = np.concatenate((a_pre_bot, a_pre_top), axis=0)
        else:
            d_vector_tendon: ndarray = d_pre_bot
            tendon_area_vector:ndarray = a_pre_bot

        d_strekk_tendon, f_strekk_tendon_vec, d_trykk_tendon, f_trykk_tendon_vec = (
            evaluate_reinforcement_from_strain(
                d_vector_tendon,
                tendon_area_vector,
                d_0,
                e_ok,
                e_s_d0,
                0,
                tendon_material,
                material,
                False,
            )
        )
        f_strekk_tendon: float = np.sum(f_strekk_tendon_vec)
        f_trykk_tendon: float = np.sum(f_trykk_tendon_vec)
        d_strekk_tendon_avg: float = np.dot(f_strekk_tendon_vec, d_strekk_tendon) / f_strekk_tendon
        
        # Må sjekke om trykk er 0 før deling
        if f_trykk_tendon == 0:
            d_trykk_tendon_avg: float = 0
        else:
            d_trykk_tendon_avg: float = np.dot(f_trykk_tendon_vec, d_trykk_tendon) / f_trykk_tendon

    else:
        f_strekk_tendon: float = 0
        f_trykk_tendon: float = 0
        d_strekk_tendon_avg: float = 0
        d_trykk_tendon_avg: float = 0

    if carbon_vector is not None:
        d_strekk_karbon, f_strekk_karbon_vec, d_trykk_karbon, f_trykk_karbon_vec = (
            evaluate_reinforcement_from_strain(
                d_karbon,
                carbon_vector,
                d_0,
                e_ok,
                e_s_d0,
                0,
                carbon_material,
                material,
                False,
            )
        )
        f_strekk_karbon: float = np.sum(f_strekk_karbon_vec)
        f_trykk_karbon: float = np.sum(f_trykk_karbon_vec)
        m_strekk_karbon: float = np.dot(f_strekk_karbon_vec, d_strekk_karbon)
        m_trykk_karbon: float = np.dot(f_trykk_karbon_vec, d_trykk_karbon)
        d_strekk_karbon_avg: float = m_strekk_karbon / f_strekk_karbon
        
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
    if sum_f_strekk_armering == 0:
        print("sum_f_ er 0")
    if sum_f_strekk_armering > 0:
        d_strekk_rebar: float = np.dot(f_strekk, d_strekk) / sum_f_strekk_armering
        d_strekk_avg: float = (d_strekk_rebar * sum_f_strekk_armering + d_strekk_tendon_avg * f_strekk_tendon + d_strekk_karbon_avg * f_strekk_karbon) / (sum_f_strekk_armering + f_strekk_tendon + f_strekk_karbon)
    else:
        d_strekk_avg: float = 0
    
    # Summerer strekkbidragene
    sum_strekk = sum_f_strekk_armering + f_strekk_tendon + f_strekk_karbon

    alpha_d: float = alpha * d_0
    e_c_uk = 0.0  # bøyning, en del vil alltid være i strekk så setter denne 0 for integralet sin del

    height_uk = height - alpha_d
    f_bet, d_alpha_d = integrate_cross_section(
        e_ok, e_c_uk, height_uk, height, material
    )  # , var_height)
    d_bet = alpha_d - d_alpha_d


    # Regner ut bidraget fra armering. Trykkmoment regnes om overkant
    sum_trykkmoment: float = np.dot(f_trykk, d_trykk) + f_trykk_tendon * d_trykk_tendon_avg + f_trykk_karbon * d_trykk_karbon_avg + f_bet * d_bet
    sum_trykk: float = sum_f_trykk_armering + f_trykk_tendon + f_trykk_karbon + f_bet 

    # d_trykk_avg er målt fra OK
    d_trykk_avg: float = abs(sum_trykkmoment / sum_trykk)

    z = d_strekk_avg - d_trykk_avg

    return alpha, sum_trykk, sum_strekk, z


def objective_function_e_s(
    e_s,
    e_c,
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
    carbon_material: CarbonMaterial = None,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    e_ok = e_c
    d0 = d_bot[0]
    alpha = e_c / (e_c - e_s)
    # e_uk = e_s / (d0 * (1 - alpha)) * (height - d0 * alpha)
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_b, f_s, z = section_integrator(
        e_ok,
        e_s,
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
        carbon_material=carbon_material,
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = abs(f_b * z)
    if abs(abs(mom_b) / max(abs(mom_s), 1e-6) - 1) < 0.001:
        print("konv")
        print("f_s: ", f_s, "f_b:", f_b)
        print("alpha:", alpha)

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_s / max(mom_b, 1e-6) - 1.0, alpha, mom_s, mom_b, z


def objective_function_e_c(
    e_s,
    e_c,
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
    carbon_material: CarbonMaterial = None,
):
    """
    Metode som kaller "calc_inner_state" og returnerer riktig versjon av M / M.
    """
    e_ok = e_c
    d0 = d_bot[0]
    alpha = e_c / (e_c - e_s)
    e_uk = e_s / (d0 * (1 - alpha)) * (height - d0)
    # Kaller funksjonen calc_inner_state for å beregne indre tilstand
    alpha, f_b, f_s, z = section_integrator(
        e_ok,
        e_uk,
        height,
        material,
        rebar_material,
        as_bot,
        as_top,
        d_bot,
        d_top,
        a_pre_bot,
        a_pre_top,
        d_pre_bot,
        d_pre_top,
        creep_eff,
        tendon_material=rebar_pre_material,
        carbon_material=carbon_material,
    )

    # Beregner momenter
    mom_s = f_s * z
    mom_b = abs(f_b * z)

    # Returnerer objektfunksjonsverdien og andre relevante verdier
    return mom_b / max(mom_s, 1e-6) - 1.0, alpha, mom_b, z


def newton_optimize_e_s(
    e_c,
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
    carbon_material: CarbonMaterial = None,
):
    """
    Metode som gjør iterasjonen for e_s via Newton-Raphson.
    Inneholder egne sjekk for e_s-optimering og er derfor skilt fra e_c.
    """
    # Initialiserer
    max_iterations = 30
    tolerance = 1e-3
    e_s = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_s, mom_b, z = objective_function_e_s(
            e_s,
            e_c,
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
            carbon_material=carbon_material,
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _, _ = objective_function_e_s(
            e_s + h,
            e_c,
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
            carbon_material=carbon_material,
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
        if e_s < 0.0 or math.isnan(e_s):
            e_s = 0.0001
            step_size *= 0.75

    # Hvis maks antall iterasjoner er nådd uten konvergens
    return -1.0, -1.0, -1.0, -1.0


def newton_optimize_e_c(
    e_s,
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
    e_cu,
    creep_eff,
    rebar_pre_material: RebarMaterial = None,
    carbon_material: CarbonMaterial = None,
):
    """
    Metode som gjør iterasjonen for e_c via Newton-Raphson.
    Inneholder egne sjekk for e_c-optimering og er derfor skilt fra e_s.
    """
    # Initialiserer
    max_iterations = 20
    tolerance = 1e-3
    e_c = initial_guess
    iterations = 0
    h = 1e-8
    step_size = 1.0

    while iterations <= max_iterations:
        iterations += 1

        # Regne ut objektfunksjonen og dens deriverte
        f_value, alpha, mom_b, z = objective_function_e_c(
            e_s,
            e_c,
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
            carbon_material=carbon_material,
        )
        abs_f_value = abs(f_value)

        f_value2, _, _, _ = objective_function_e_c(
            e_s,
            e_c + h,
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
            carbon_material=carbon_material,
        )

        f_prime = (f_value2 - f_value) / h

        # Justerer verdi i henhold til Newton-Raphson-iterasjonen
        if abs_f_value > 1e-6:
            e_c -= step_size * f_value / f_prime

            # Sjekk om tøyningen er altfor stor
            if e_c < e_cu:
                e_c = e_cu
                step_size *= 0.5  # Kan kanskje på sikt fjerne step_size.

        # Sjekk mot konvergenskriteriet
        if abs_f_value < tolerance:
            return e_c, alpha, mom_b, z

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
    area_carbon: ndarray = None,
    d_carbon: ndarray = None,
    carbon_material: CarbonMaterial = None,
    creep_eff: float = 0,
) -> float:
    """For ULS"""
    # Må ha en initiell testverdi
    initial_guess = 0.015
    e_cu = concrete_material.get_e_cu()
    e_c = e_cu

    # Kalkulasjon starter
    # Må finne den verdien/kombinasjonen av e_cu og e_s som gir indre likevekt i tverrsnittet.
    # Starter med å anta e_c = e_cu og ser om ulike verdier av e_s kan gi likevekt.

    sum_as_bot = np.sum(as_area_bot)
    sum_as_top = np.sum(as_area_top)

    alpha, mom_b, z = 0, 0, 0
    if sum_as_bot > sum_as_top:
        e_s, alpha, mom_b, z = newton_optimize_e_s(
            e_c,
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
            rebar_pre_material=rebar_material,
            carbon_material=carbon_material,
        )
    else:
        e_s = -1.0

    print("tester--------------")
    # Sjekker om første iterasjon var vellykket
    if e_s == -1.0:
        # Betongtøyningen kan ikke nå e_cu. Setter en armeringstøyning og finner
        # betongtøyning som gir likevekt i tverrsnittet (e_c < e_cu).
        e_s = initial_guess
        initial_guess = -0.00117
        e_c, alpha, mom_b, z = newton_optimize_e_c(
            e_s,
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
            e_cu,
            creep_eff,
            rebar_pre_material,
            carbon_material,
        )

    return (alpha, mom_b, e_s, e_c, z)


def get_width(_a: float, _b: float) -> float:
    """Dummy, men fungerer for konstant bredde"""
    return 200.0


def get_width2(height_i: float, var: float) -> float:
    """Lager en funksjon som beskriver bredden for enhver høyde"""
    if height_i < 80:
        return 320
    if height_i < 220:
        return 320 - 220 * (height_i - 80) / 140
    if height_i < 220 + var:
        return 100
    if height_i < 220 + var + 50:
        rel_height = height_i - (220 + var)
        return 100 + 320 * (height_i - rel_height) / 50

    return 420


if __name__ == "__main__":
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    armering: RebarMaterial = RebarB500NC()
    spennarmering: RebarMaterial = Tendon()
    spennarmering.prestressd_to(25)
    antall_vector = np.array([2, 2])
    area_vector = spennarmering.get_area(antall_vector)
    

    height = 300
    as_area_bot = np.array([228 * 3.14, 64 * 3.14])
    as_area_top = np.array([64 * 3.14])
    d_bot = np.array([250, 200])
    d_top = np.array([50])
    d_pre_bot = np.array([250, 200])
    alpha, mom, e_s, e_c, z = integration_iterator_ultimate(height, as_area_bot, as_area_top, d_bot, d_top, betong_b35, armering, rebar_pre_material=spennarmering, a_pre_bot=area_vector, d_pre_bot=d_pre_bot)
    print("alpha:", alpha, "mom:", mom / 1e6, "e_c:", e_c, "e_s:", e_s)


def testing_strain_part():
    """Brukt for å teste evaluate reinforcement blant annet"""
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    armering: RebarMaterial = RebarB500NC()
    spennarmering: RebarMaterial = Tendon()
    spennarmering.prestressd_to(115)
    # sum_f, sum_m, d_bet = integrate_cross_section(0.0035, 0, 0, 200, betong_b35, 300)
    # print(f"Force is {sum_f / 1000:.1f} kN")

    # height = np.array(300)
    height = 300
    as_area_bot = np.array([228 * 3.14, 64 * 3.14])
    as_area_top = np.array([64 * 3.14])
    d_bot = np.array([250, 200])
    d_top = np.array([50])
    alpha, mom, e_s, e_c, z = integration_iterator_ultimate(
        height, as_area_bot, as_area_top, d_bot, d_top, betong_b35, armering
    )
    # print("alpha:", alpha, ". e_c:", e_c, "e_s:", e_s)
    # e_c_uk: float = 0.0026340921430255495

    d_strekk, f_strekk, d_trykk, f_trykk = evaluate_reinforcement_from_strain(
        np.array([250, 200, 50]),
        np.array([228 * 3.14, 64 * 3.14, 64 * 3.14]),
        d_bot[0],
        -0.0035,
        e_s,
        0,
        armering,
        betong_b35,
        True,
    )
    alpha_d = alpha * d_bot[0]
    f_bet, d_bet = integrate_cross_section(
        e_c, 0, height - alpha_d, height, betong_b35
    )  # , var_height)

    print("f_bet:", f_bet)
    print("Trykk:", f_trykk.sum())
    print("Strekk:", f_strekk.sum())

    alpha, sum_trykk, sum_strekk_arm, z = section_integrator(
        -0.0035,
        e_s,
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
        betong_b35.get_e_cu(),
        0.002,
        0,
        spennarmering,
        betong_b35,
        True,
    )
    print(f"f_strekk_f: {f_strekk_f}. f_trykk_f: {f_trykk_f}")