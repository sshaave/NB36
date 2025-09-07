"""Beregningsprogram for fritt opplagt betongbjelke med jevnt fordelt last

Dette programmet regner på fritt opplagte bjelker med jevnt fordelt last. Betongbjelkens tverrsnitt kan være av alle,
og må defineres ved bruk av "width_function" rett over main-funksjonen. Hvis bjelken har varierende tverrsnittshøyde (SDT),
må height være en vektor for som har samme størrelse som momentvektor.

Materialer:
 - betong med og uten strekkfasthet (strekkfasthet kun i SLS)
 - vanlig slakkarmering B500NC eller B400NC
 - spennarmering (pass på spennkraft, programmet setter den til 0 aka forblender den, hvis likevekt ikke oppnås i gitt snitt)
 - karbonfiber

 På grunn av programmet er satt opp for å ha strekksone kun i UK kan ikke kontinuerlige bjelker regnes på.

 For andre typer laster (punktlaster) må brukeren selv sette opp momentvektorene.
"""

import numpy as np
from numpy import ndarray
from hjelpemetoder import (
    calc_deflection_with_curvatures,
    find_eps_carbon,
    get_moments_simply_supported_beam,
)
from materialmodeller import (
    CarbonFiber,
    CarbonMaterial,
    ConcreteMaterial,
    RebarMaterial,
    RebarB400NC,
    RebarB500NC,
    Tendon,
)

from tverrsnittsberegninger import (
    Tverrsnitt,
    find_equilibrium_strains,
    integration_iterator_ultimate,
)


def width_function(height_i: float, height_snitt: float = 0) -> float:
    """Lager en funksjon som beskriver bredden for enhver høyde
    height_snitt: total høyde i betraktet snitt
    height_i: betraktet høyde hvor bredde skal leses av
    """
    top_rectangle = 150
    top_cone = 50
    bot_rectangle = 80
    bot_cone = 140
    if height_i < bot_rectangle or height_snitt == 0:
        # Nederste rektangel
        return 320
    if height_i < bot_rectangle + bot_cone:
        # Nederste kjegle
        return 320 - 220 * (height_i - bot_rectangle) / bot_cone
    if height_i >= height_snitt - top_rectangle:
        # Øverste rektangel
        return 420
    if height_i > height_snitt - top_rectangle - top_cone:
        # Øverste kjegle
        rel_height = height_snitt - top_rectangle - height_i
        return 420 - 320 * rel_height / top_cone
    
    # Midt på
    return 100
    


if __name__ == "__main__":
    ### INPUT STARTER ###
    # Bjelkens lengde
    bjelkelengde: float = 24.27  # i m

    # Betongparametere
    f_ck: float = 45                            # Betongkvalitet
    materialmodell = "Parabola"                 # kan velge mellom Parabola og Sargin (pass på)
    strekkfast_betong_SLS: bool = False          # Velg om strekkfasthet skal inkluderes for SLS-beregninger

    # Definerer tverrsnitt. Høyde (total høyde) kan være en vektor og bredde kan være et tall eller en funksjon.
    # For et I-tverrsnitt må totalhøyde være en vektor og bredde en funksjon av høyden
    # Høyde-, bredde, og momentvektorer må ha lik lengde. Hvis det er kompakte tverrsnitt i endene (SDT) må snittene være
    height = np.concatenate([np.linspace(791, 1600, 42), np.linspace(1600, 791, 42)[1:]])

    # Bredde kan være et tall eller en funksjon (funksjon er nødvendig for I-tverrsnitt)
    width = width_function
    kompakt_lengde = 1000                       # SDT har kompakte tverrsnitt i hver ende, skriv inn lengden med dette (kan være 0)

    # Antall snitt langs bjelken
    antall_punkter_standard: int = 11           # benyttes hvis høyde er ett tall. Kan endres på av bruker
    antall_punkter: int = len(height) if isinstance(height, (list, ndarray)) else antall_punkter_standard

    # Definerer vanlig armering
    armeringskvalitet: str = "B400NC"           # B400NC valgt
    as_area_bot = np.array([])                  # ingen i UK
    as_area_top = np.array([200 * np.pi])       # 2ø16 + 2ø12 i øverste lag
    d_bot = np.array([])                        # Måles fra UK betong
    d_top = np.array([40])                      # Måles fra OK betong

    # Definerer spennarmering. Kablene har areal på 100mm2
    forspenningskraft: float = 80.0             # Forspenningskraft i kN. Pass på, for mye forspenning gir forblending i evaluert snitt
    antall_vektor_ok = np.array([2])
    antall_vektor_uk = np.array([4, 6, 4, 2])
    d_pre_bot = np.array([40, 80, 120, 160])    # Fra UK betong
    d_pre_top = np.array([40])                  # Fra OK betong
    print_forspenningskraft = False             # Rappoterer forblending eller redusert i kraft for å finne likevekt (strekk OK ikke implementert)

    # Definerer karbonfiber
    # I eksempelet her er det valgt 50 mm bredde, 1.4 mm tykkelse, 1 på hver side -> 2 stk
    a_carbon: ndarray = np.array([50 * 2 * 1.4])
    d_carbon: ndarray = np.array([40])          # Målt fra UK betong

    # Linjelaster - bruker må legge inn evt egenvekt av bjelke
    q_uls: float = 41.5                         # Linjelast i ULS
    q_sls: float = 6 + 11.25                    # Linjelast i SLS
    q_montering: float = 1                      # Linjelast i bjelke når fiber monteres

    # Svinntøyning og effektivt kryptall
    eps_svinn_promille: float = -0.0            # -0.01 eksempelverdi
    eps_svinn: float = eps_svinn_promille / 1e3
    creep_eff: float = 1.6                      # 1.61 eksempelverdi

    #### INPUT FERDIG ####

    ### Initialisering ###

    # Momentverdier langs bjelken i det karbonfiberen monteres
    moment_vector_uls: ndarray = get_moments_simply_supported_beam(
        q_uls, bjelkelengde, num_points=antall_punkter
    )
    moment_vector_sls: ndarray = get_moments_simply_supported_beam(
        q_sls, bjelkelengde, num_points=antall_punkter
    )
    moment_vector_montering: ndarray = get_moments_simply_supported_beam(
        q_montering, bjelkelengde, num_points=antall_punkter
    )

    moment_max_uls: float = moment_vector_uls.max()
    m_max_montering: float = moment_vector_montering.max()

    # Sjekker form på input
    assert len(as_area_bot) == len(d_bot), (
        "Feil i input for vanlig armering UK. A- og d-vektor må være like lange"
    )
    assert len(as_area_top) == len(d_top), (
        "Feil i input for vanlig armering OK. A- og d-vektor må være like lange"
    )
    assert len(antall_vektor_uk) == len(d_pre_bot), (
        "Feil i input for spennarmering UK. A- og d-vektor må være like lange"
    )
    assert len(antall_vektor_ok) == len(d_pre_top), (
        "Feil i input for spennarmering OK. A- og d-vektor må være like lange"
    )
    assert len(a_carbon) == len(d_carbon), (
        "Feil i input for karbonfiber. A- og d-vektor må være like lange"
    )

    # Lager materialer hvis det er definert arealer og avstander er definert
    # Betong
    betong: ConcreteMaterial = ConcreteMaterial(f_ck, materialmodell)
    betong.set_creep(creep_eff)
    if not strekkfast_betong_SLS:
        betong.set_f_ctm_to_0()
    
    # Slakkarmering
    if (sum(d_bot) > 0 and sum(as_area_bot) > 0) or (
        sum(d_top) > 0 and sum(as_area_top) > 0
    ):
        armering: RebarMaterial = (
            RebarB500NC() if armeringskvalitet == "B500NC" else RebarB400NC()
        )
    else:
        armering, as_area_bot, as_area_top = None, np.array([]), np.array([])
        d_bot, d_top = np.array([]), np.array([])

    # Spennarmering føroppspent
    if (sum(d_pre_bot) > 0 and sum(antall_vektor_uk) > 0) or (
        sum(d_pre_top) > 0 and sum(antall_vektor_ok) > 0
    ):
        spennarmering: RebarMaterial = Tendon()
        spennarmering.set_fp(forspenningskraft)
        area_vector_ok = spennarmering.get_area(antall_vektor_ok)
        area_vector_uk = spennarmering.get_area(antall_vektor_uk)
    else:
        spennarmering, area_vector_ok, area_vector_uk = None, np.array([]), np.array([])
        d_pre_bot, d_pre_top = np.array([]), np.array([])

    # Karbonfiber
    if sum(a_carbon) > 0 and sum(d_carbon) > 0:
        karbonfiber: CarbonMaterial = CarbonFiber()
    else:
        karbonfiber, a_carbon, d_carbon = None, np.array([]), np.array([])

    # Lagrer tverrsnittobjektet
    # Starter med å lage høydevektor hvis kun ett tall er gitt
    if isinstance(height, (float, int)):
        height_vector = np.full(antall_punkter, height)
    else:
        height_vector = height

    tverrsnitt: Tverrsnitt = Tverrsnitt(
        height_vector,
        width,
        as_area_bot,
        as_area_top,
        d_bot,
        d_top,
        a_pre_bot=area_vector_uk,
        d_pre_bot=d_pre_bot,
        a_pre_top=area_vector_ok,
        d_pre_top=d_pre_top,
        a_carbon=a_carbon,
        d_carbon=d_carbon,
    )
    # For bjelker med kompakte ender
    tverrsnitt.set_lengde_for_kompakt(kompakt_lengde)

    # -- Initialisering ferdig -- #

    # ---------- ULS ------------ #

    # Forenkler og bruker maksimal høyde for tverrsnittet og maks moment
    tverrsnitt.set_height_to_max()
    # Finner likevekt i mest belastet snitt for å finne differansetøyning i bjelke og karbonfiber
    if karbonfiber is not None:
        is_ck_not_cd: bool = True  # starter med bruks
        eps_ok, eps_uk, _, _ = find_equilibrium_strains(
            1000 * m_max_montering,
            betong,
            tverrsnitt,
            armering,
            spennarmering,
            is_ck_not_cd=is_ck_not_cd,
        )

        eps_carbon = find_eps_carbon(eps_ok, eps_uk, tverrsnitt)
        print(f"Geometrisk tøyning ved høyden karbonfiberen monteres i, ved montasjelast: {eps_carbon:.7f}") # Karbonfiberen har 0 tøyning ved denne geometriske tøyningen
        karbonfiber.set_eps_s_0_state(eps_carbon)

    # Regner ut momentkapasitet i ULS (differanse i tverrsnitt og karbonfiber hensyntatt) med maks moment og største tverrsnittshøyde
    print("--------------- ULS ---------------")
    alpha_uls, mom_kapasitet, eps_s, eps_c, z = integration_iterator_ultimate(
        tverrsnitt,
        betong,
        rebar_material=armering,
        rebar_pre_material=spennarmering,
        carbon_material=karbonfiber,
    )

    # Printer resultater fra ULS-beregning
    print(
        f"Momentkapasitet: {mom_kapasitet / 1e6:.1f} kNm, betongtøyning: {eps_c:.6f}, armeringstøyning: {eps_s:.6f}"
    )
    if alpha_uls > 0.617:
        print(f"Alpha: {alpha_uls:.3f} -> Overarmert tverrsnitt")
    else:
        print(f"Alpha: {alpha_uls:.3f} -> Underarmert tverrsnitt")

    # ---------- SLS ------------ #
    print("--------------- SLS ---------------")

    # Finner deformasjoner, kurvaturer, og rotasjoner
    curvatures, rotations, deflections, max_deflection = (
        calc_deflection_with_curvatures(
            moment_vector_sls,
            moment_vector_montering,
            bjelkelengde,
            tverrsnitt,
            betong,
            armering,
            spennarmering,
            karbonfiber,
            eps_svinn,
            True,
            print_forspenningskraft,
        )
    )

    # Printer største nedbøyning
    print(f"Største nedbøyning: {np.round(max_deflection, 2)} mm")
