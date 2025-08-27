import numpy as np
from numpy import ndarray
from hjelpemetoder import calc_deflection_with_curvatures, find_eps_carbon, get_moments_simply_supported_beam
from materialmodeller import (
    CarbonFiber,
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    RebarB400NC,
    Tendon,
)

from tverrsnittsberegninger import Tverrsnitt, find_equilibrium_strains, integration_iterator_ultimate

if __name__ == "__main__":
    # Definerer materialer
    betong_b45: ConcreteMaterial = ConcreteMaterial(45, material_model="Parabola")
    
    armering: RebarMaterial = RebarB400NC() # armering: RebarMaterial = RebarB500NC()
    karbonfiber: CarbonMaterial = CarbonFiber()
    
    # Definerer tverrsnitt
    height = 450

    # Definerer vanlig armering
    as_area_bot = np.array([5 * 64 * np.pi])  # 5 stk 16mm
    as_area_top = np.array([5 * 64 * np.pi])
    d_bot = np.array([400])
    d_top = np.array([50])

    # Definerer spennarmering
    spennarmering: RebarMaterial = Tendon()
    spennarmering.set_fp(0)
    antall_vector_ok = np.array([0])
    antall_vector_uk = np.array([0])# np.array([4, 6, 4, 2])
    area_vector_ok = spennarmering.get_area(antall_vector_ok)
    area_vector_uk = spennarmering.get_area(antall_vector_uk)
    d_pre_bot = np.array([height - 40, height - 80]) #height - 80, height - 120, height - 160])
    d_pre_top = np.array([40])
    spennarmering = None

    # Definerer karbonfiber
    # 50 mm bredde, 1.4 mm tykkelse, 1 på hver side -> 2 stk
    carbon_vector: ndarray = np.array([50 * 1.4 * 11])
    d_carbon: ndarray = np.array([height - 40])
    
    # Linjelaster
    q_uls: float = 20
    q_sls: float = 10
    q_montering: float = 5 # montering av karbonfiber
    
    # Lagrer tverrsnittobjektet
    tverrsnitt: Tverrsnitt = Tverrsnitt(height, as_area_bot, as_area_top, d_bot, d_top,
                             #a_pre_bot=area_vector_uk, d_pre_bot=d_pre_bot,
                             #a_pre_top=area_vector_ok, d_pre_top=d_pre_top,
                             a_carbon=carbon_vector, d_carbon=d_carbon,)
    
    # Momentverdier langs bjelken i det karbonfiberen monteres
    # Moment for simply supported beam with UDL: M(x) = q * x * (L - x) / 2
    beam_length: float = 5  # i m
    moment_vector_uls: ndarray = get_moments_simply_supported_beam(q_uls, beam_length, num_points=15)
    moment_vector_sls: ndarray = get_moments_simply_supported_beam(q_sls, beam_length, num_points=15)
    moment_vector_montering: ndarray = get_moments_simply_supported_beam(q_montering, beam_length, num_points=15)
    
    moment_max_uls: float = moment_vector_uls.max()


    # TODO! lag funksjon som henter moment fra linjelast
    
    # Svinntøyning og effektivt kryptall
    eps_svinn: float = .09 # 0.01 % eksempelverdi
    creep_eff: float = 0.#1.61  # eksempelverdi
    
    
    # ------ ULS --------
    # Finner likevekt i mest belastet snitt for å finne differansetøyning i bjelke og karbonfiber
    is_ck_not_cd: bool = True  # starter med bruks
    eps_ok, eps_uk, _, _ = find_equilibrium_strains(moment_max_uls / 3, betong_b45, tverrsnitt, armering,
                                                    spennarmering, is_ck_not_cd=is_ck_not_cd)
    eps_carbon = find_eps_carbon(eps_ok, eps_uk, tverrsnitt)
    print(f"eps_carbon: {eps_carbon:.7f}")
    #karbonfiber.set_eps_s_0_state(eps_carbon)
    
    # Regner ut momentkapasitet i ULS (differanse i tverrsnitt og karbonfiber hensyntatt)
    alpha_uls, mom_kapasitet, eps_s, eps_c, z = integration_iterator_ultimate(
        tverrsnitt, betong_b45, rebar_material=armering, rebar_pre_material=spennarmering,
        carbon_material=karbonfiber, creep_eff=creep_eff)
    
    print(f"alpha: {alpha_uls:.3f}, mom: {mom_kapasitet/1e6:.1f} kNm, eps_c: {eps_c:.6f}, eps_s: {eps_s:.6f}")


    # ------ SLS --------
    bjelkelengde: float = 5 # i m
    creep_eff: float = 0. # endrer fra 0
    karbonfiber.reset_0_state()
    curvatures, rotations, deflections, max_deflection = calc_deflection_with_curvatures(moment_vector_sls, bjelkelengde, tverrsnitt,
                                                  betong_b45, rebar_material=armering,
                                                  tendon_material=spennarmering,
                                                  carbon_material=karbonfiber,
                                                  eps_cs=eps_svinn, creep_eff=creep_eff)
    #print(np.round(deflections, 2))
    print(np.round(max_deflection, 2))
    