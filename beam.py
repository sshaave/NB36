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

def width_function(height_i: float, height: float = 0) -> float:
    """Lager en funksjon som beskriver bredden for enhver høyde
        height_i: betraktet høyde hvor bredde skal leses av
        height: total høyde på tverrsnittet for gitt snitt
    """
    if height_i < 80:
        return 320
    if height_i < 220:
        return 320 - 220 * (height_i - 80) / 140
    if height_i < 220 + height:
        return 100
    if height_i < 220 + height + 50:
        rel_height = height_i - (220 + height) # Må fikses
        return 100 + 320 * rel_height / 50
    return 420

if __name__ == "__main__":
    # Definerer materialer
    f_ck: float = 45  # Betongkvalitet
    betong: ConcreteMaterial = ConcreteMaterial(f_ck, material_model="Parabola")  # kan velge mellom Parabola og Sargin (pass på)
    
    # Bjelkens lengde
    bjelkelengde: float = 5  # i m
    
    # Definerer tverrsnitt. Høyde (total høyde) er en vektor og bredde kan være et tall eller en funksjon.
    # For et I-tverrsnitt må totalhøyde være vektor og bredde en funksjon av høyden
    # Høyden må ha like mange datapunkt som momentvektoren
    height = 350
    width = width_function # width = 300 er også et alternativ
    antall_punkter_standard: int = 15 # benyttes hvis høyde er et tall. Kan endres på av bruker
    antall_punkter: int = len(height) if isinstance(height, (list, ndarray)) else antall_punkter_standard

    # Definerer vanlig armering
    armerings_kvalitet: str = "B500NC" # B400NC eller B500NC
    as_area_bot = np.array([2 * 64 * np.pi, 100])  # 5 stk 16mm
    as_area_top = np.array([2 * 64 * np.pi, 50])
    d_bot = np.array([389, 360])
    d_top = np.array([61, 88])

    # Definerer spennarmering. Kablene har areal på 100mm2
    forspenningskraft: float = 0. # Forspenningskraft i kN
    antall_vektor_ok = np.array([0])
    antall_vektor_uk = np.array([2])# np.array([4, 6, 4, 2])
    d_pre_bot = np.array([410]) #height - 80, height - 120, height - 160])
    d_pre_top = np.array([40])

    # Definerer karbonfiber
    # I eksempelet her er det valgt 50 mm bredde, 1.4 mm tykkelse, 1 på hver side -> 2 stk
    a_carbon: ndarray = np.array([20 * 1.4 * 2])
    d_carbon: ndarray = np.array([height - 40])
    
    # Linjelaster - husk å ta hensyn til egenvekt av bjelke
    q_uls: float = 30
    q_sls: float = 6 + 11.25
    q_montering: float = 1 # ved montering av karbonfiber
    
    # Svinntøyning og effektivt kryptall
    eps_svinn_promille: float = -0.0 # -0.01 % eksempelverdi
    eps_svinn: float = eps_svinn_promille / 1000
    creep_eff: float = 1.6 #1.61  # eksempelverdi
    
    ### INPUT FERDIG ###
    
    
    # Momentverdier langs bjelken i det karbonfiberen monteres
    moment_vector_uls: ndarray = get_moments_simply_supported_beam(q_uls, bjelkelengde, num_points=antall_punkter)
    moment_vector_sls: ndarray = get_moments_simply_supported_beam(q_sls, bjelkelengde, num_points=antall_punkter)
    moment_vector_montering: ndarray = get_moments_simply_supported_beam(q_montering, bjelkelengde, num_points=antall_punkter)
    
    moment_max_uls: float = moment_vector_uls.max()
    m_max_montering: float = moment_vector_montering.max()

    # Lager materialer hvis det er definert arealer og avstander er definert
    ##########################################
    betong.set_creep(creep_eff)
    if (sum(d_bot) > 0 and sum(as_area_bot) > 0) or (sum(d_top) > 0 and sum(as_area_top) > 0):
        if armerings_kvalitet == "B500NC":
            armering: RebarMaterial = RebarB500NC()
        else:
            armering: RebarMaterial = RebarB400NC()
    else: 
        armering = None
        as_area_bot, as_area_top = np.array([]), np.array([])
        d_bot, d_top = np.array([]), np.array([])
        
    if (sum(d_pre_bot) > 0 and sum(antall_vektor_uk) > 0) or (sum(d_pre_top) > 0 and sum(antall_vektor_ok) > 0):
        spennarmering: RebarMaterial = Tendon()
        spennarmering.set_fp(forspenningskraft)
        area_vector_ok = spennarmering.get_area(antall_vektor_ok)
        area_vector_uk = spennarmering.get_area(antall_vektor_uk)
    else:
        spennarmering = None
        area_vector_ok, area_vector_uk = np.array([]), np.array([])
        d_pre_bot, d_pre_top = np.array([]), np.array([])
    
    if sum(a_carbon) > 0 and sum(d_carbon) > 0:
        karbonfiber: CarbonMaterial = CarbonFiber()
    else:
        karbonfiber = None
        a_carbon, d_carbon = np.array([]), np.array([])
    ##########################################
    # Lagrer tverrsnittobjektet
    tverrsnitt: Tverrsnitt = Tverrsnitt(height, width, as_area_bot, as_area_top, d_bot, d_top,
                             a_pre_bot=area_vector_uk, d_pre_bot=d_pre_bot,
                             a_pre_top=area_vector_ok, d_pre_top=d_pre_top,
                             a_carbon=a_carbon, d_carbon=d_carbon)
    
    ##########################################    
    # -------- ULS ---------
    # Forenkler og bruker maksimal høyde for tverrsnittet og maks moment
    height_original = tverrsnitt.get_height()
    height_max: float | int = height_original if isinstance(height_original, (float, int)) else height_original.max()
    tverrsnitt.height = height_max
    
    # Finner likevekt i mest belastet snitt for å finne differansetøyning i bjelke og karbonfiber
    if karbonfiber is not None:
        is_ck_not_cd: bool = True  # starter med bruks
        eps_ok, eps_uk, _, _ = find_equilibrium_strains(1000 * m_max_montering, betong, tverrsnitt, armering,
                                                        spennarmering, is_ck_not_cd=is_ck_not_cd)
    
        eps_carbon = find_eps_carbon(eps_ok, eps_uk, tverrsnitt)
        print(f"eps_carbon: {eps_carbon:.7f}")
        karbonfiber.set_eps_s_0_state(eps_carbon)
    
    # Regner ut momentkapasitet i ULS (differanse i tverrsnitt og karbonfiber hensyntatt) med maks moment og største tverrsnittshøyde    
    alpha_uls, mom_kapasitet, eps_s, eps_c, z = integration_iterator_ultimate(
        tverrsnitt, betong, rebar_material=armering, rebar_pre_material=spennarmering,
        carbon_material=karbonfiber)
    
    # Tverrsnitt får originale høydeverdier igjen
    tverrsnitt.height = height_original
    
    # Printer resultater fra ULS-beregning
    print("---- ULS ----")
    print(f"alpha: {alpha_uls:.3f}, mom: {mom_kapasitet/1e6:.1f} kNm, eps_c: {eps_c:.6f}, eps_s: {eps_s:.6f}")

    ##########################################
    # ------ SLS ------------
    print("---- SLS ----")
    curvatures, rotations, deflections, max_deflection = calc_deflection_with_curvatures(
        moment_vector_sls, moment_vector_montering, bjelkelengde, tverrsnitt, betong,
        rebar_material=armering, tendon_material=spennarmering, carbon_material=karbonfiber, eps_cs=eps_svinn)
    print(f"Største deformasjon: {np.round(max_deflection, 2)} mm")
    