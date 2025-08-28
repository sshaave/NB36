import numpy as np
from numpy import ndarray
from hjelpemetoder import calc_deflection_with_curvatures, find_eps_carbon, get_moments_simply_supported_beam
from materialmodeller import (
    CarbonFiber,
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    RebarB400NC,
    RebarB500NC,
    Tendon,
)

from tverrsnittsberegninger import Tverrsnitt, find_equilibrium_strains, integration_iterator_ultimate

if __name__ == "__main__":
    # Definerer materialer
    f_ck: float = 45  # Betongkvalitet
    betong: ConcreteMaterial = ConcreteMaterial(f_ck, material_model="Parabola")  # kan velge mellom Parabola og Sargin (pass på)
    
    # Bjelkens lengde
    bjelkelengde: float = 5  # i m
    
    # Definerer tverrsnitt
    height = 400

    # Definerer vanlig armering
    armerings_kvalitet: str = "B500NC" # B400NC eller B500NC
    as_area_bot = np.array([5 * 64 * np.pi])  # 5 stk 16mm
    as_area_top = np.array([2 * 64 * np.pi])
    d_bot = np.array([400])
    d_top = np.array([50])

    # Definerer spennarmering. Kablene har areal på 100mm2
    forspenningskraft: float = 0. # Forspenningskraft i kN
    antall_vektor_ok = np.array([0])
    antall_vektor_uk = np.array([0])# np.array([4, 6, 4, 2])
    d_pre_bot = np.array([height - 40, height - 80]) #height - 80, height - 120, height - 160])
    d_pre_top = np.array([40])

    # Definerer karbonfiber
    # I eksempelet her er det valgt 50 mm bredde, 1.4 mm tykkelse, 1 på hver side -> 2 stk
    a_carbon: ndarray = np.array([50 * 1.4 * 0])
    d_carbon: ndarray = np.array([height - 40])
    
    # Linjelaster
    q_uls: float = 20
    q_sls: float = 10
    q_montering: float = 5 # ved montering av karbonfiber
    
    # Lagrer tverrsnittobjektet. Inkluder relevante arealer og d
    tverrsnitt: Tverrsnitt = Tverrsnitt(height, as_area_bot, as_area_top, d_bot, d_top,
                             #a_pre_bot=area_vector_uk, d_pre_bot=d_pre_bot,
                             #a_pre_top=area_vector_ok, d_pre_top=d_pre_top,
                             a_carbon=a_carbon, d_carbon=d_carbon,)
    
    # Svinntøyning og effektivt kryptall
    eps_svinn_promille: float = -0.09 # -0.01 % eksempelverdi
    eps_svinn: float = eps_svinn_promille / 1000
    creep_eff: float = 0. #1.61  # eksempelverdi
    
    ### INPUT FERDIG ###
    
    
    # Momentverdier langs bjelken i det karbonfiberen monteres
    moment_vector_uls: ndarray = get_moments_simply_supported_beam(q_uls, bjelkelengde, num_points=15)
    moment_vector_sls: ndarray = get_moments_simply_supported_beam(q_sls, bjelkelengde, num_points=15)
    moment_vector_montering: ndarray = get_moments_simply_supported_beam(q_montering, bjelkelengde, num_points=15)
    
    moment_max_uls: float = moment_vector_uls.max()
    m_max_montering: float = moment_vector_montering.max()

    # Lager materialer hvis det er definert arealer og avstander er definert
    ##########################################
    betong.set_creep(creep_eff)
    if (sum(d_bot) > 0 and sum(as_area_bot) > 0) or (sum(d_top) > 0 and sum(as_area_top) > 0):
        if armerings_kvalitet == "B500NC":
            armering: RebarMaterial = RebarB500NC()
        else:
            armering: RebarMaterial = RebarB400NC() # armering: RebarMaterial = RebarB500NC()
    else: 
        armering = None
        
    if (sum(d_pre_bot) > 0 and sum(antall_vektor_uk) > 0) or (sum(d_pre_top) > 0 and sum(antall_vektor_ok) > 0):
        spennarmering: RebarMaterial = Tendon()
        spennarmering.set_fp(forspenningskraft)
        area_vector_ok = spennarmering.get_area(antall_vektor_ok)
        area_vector_uk = spennarmering.get_area(antall_vektor_uk)
    else:
        spennarmering = None
    
    if sum(a_carbon) > 0 and sum(d_carbon) > 0:
        karbonfiber: CarbonMaterial = CarbonFiber()
    else:
        karbonfiber = None
    ##########################################
        
    # ------ ULS --------
    # Finner likevekt i mest belastet snitt for å finne differansetøyning i bjelke og karbonfiber
    if karbonfiber is not None:
        is_ck_not_cd: bool = True  # starter med bruks
        eps_ok, eps_uk, _, _ = find_equilibrium_strains(1000 * m_max_montering, betong, tverrsnitt, armering,
                                                        spennarmering, is_ck_not_cd=is_ck_not_cd)
    
        eps_carbon = find_eps_carbon(eps_ok, eps_uk, tverrsnitt)
        print(f"eps_carbon: {eps_carbon:.7f}")
        karbonfiber.set_eps_s_0_state(eps_carbon)
    
    # Regner ut momentkapasitet i ULS (differanse i tverrsnitt og karbonfiber hensyntatt)
    alpha_uls, mom_kapasitet, eps_s, eps_c, z = integration_iterator_ultimate(
        tverrsnitt, betong, rebar_material=armering, rebar_pre_material=spennarmering,
        carbon_material=karbonfiber)
    
    print(f"alpha: {alpha_uls:.3f}, mom: {mom_kapasitet/1e6:.1f} kNm, eps_c: {eps_c:.6f}, eps_s: {eps_s:.6f}")


    # ------ SLS --------
    bjelkelengde: float = 5 # i m

    if karbonfiber is not None:
        karbonfiber.reset_0_state()
    curvatures, rotations, deflections, max_deflection = calc_deflection_with_curvatures(moment_vector_sls, bjelkelengde, tverrsnitt,
        betong, rebar_material=armering, tendon_material=spennarmering, carbon_material=karbonfiber, eps_cs=eps_svinn)
    #print(np.round(deflections, 2))
    print(np.round(max_deflection, 2))
    