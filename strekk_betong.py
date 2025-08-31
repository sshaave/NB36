from typing import Tuple

from tverrsnitt import Tverrsnitt
from materialmodeller import ConcreteMaterial


def beregn_strekk_betong(material: ConcreteMaterial, ts: Tverrsnitt, alpha_d: float, eps_uk: float, eps_s_u: float) -> Tuple[float, float]:
    """Beregner strekkbidrag fra betongen i snittet. Returnerer kraft og posisjon."""
    # Forenklet modell med konstant strekkspenning i hele strekksonen
    height_snitt: float = ts.get_height_i()
    height_tension: float = height_snitt - alpha_d
    f_ctm: float = material.get_f_ctm()
    e_cm: float = material.get_e_cm()
    eps_ctm: float = f_ctm / e_cm
    assert eps_uk > 0, "eps_uk må være positiv for å ha strekk i betongen"
    
    iterations = 50
    delta_eps: float = eps_uk / iterations
    delta_h: float = height_tension / iterations
    sum_f, sum_mom = 0.0, 0.0
    for i in range(1, iterations):
        height_i = delta_h * (i + 0.5)  # mm
        width_i = ts.get_width(height_i)
        area: float = height_i * width_i
        
        eps_i = delta_eps * (i - 0.5)
        spenning: float = bilinear_tension_model(eps_i, e_cm, f_ctm, eps_ctm, eps_s_u)
        f_i = area * spenning
        sum_f += f_i
        sum_mom += f_i * (height_i + )
    
    
    d_s_bet = d_strekk_avg / 2  # mm, midt i strekksonen
    return f_s_bet, d_s_bet

def bilinear_tension_model(eps: float, e_cm: float, f_ctm: float, eps_ctm: float, eps_s_u) -> float:
    """Bilineær materialmodell for betong i strekk. Returnerer spenning i MPa."""
    if eps <= 0:
        return 0.0
    elif eps < eps_ctm:
        return e_cm * eps  # Lineær opp til f_ctm
    elif eps < eps_s_u:
        return f_ctm * (eps_s_u - eps) / (eps_s_u - eps_ctm)  # Lineær ned til 0 ved 2*eps_ctm
    else:
        return 0.0