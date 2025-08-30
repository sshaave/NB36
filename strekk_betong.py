from typing import Tuple

from tverrsnitt import Tverrsnitt
from materialmodeller import ConcreteMaterial


def beregn_strekk_betong(material: ConcreteMaterial, ts: Tverrsnitt, alpha_d: float, eps_uk: float) -> Tuple[float, float]:
    """Beregner strekkbidrag fra betongen i snittet. Returnerer kraft og posisjon."""
    # Forenklet modell med konstant strekkspenning i hele strekksonen
    height_i: float = ts.get_height_i()
    f_ctm: float = material.get_f_ctm()
    eps_ctm: float = f_ctm / material.get_e_cm()
    assert eps_uk > 0, "eps_uk må være positiv for å ha strekk i betongen"
    
    
    d_s_bet = d_strekk_avg / 2  # mm, midt i strekksonen
    return f_s_bet, d_s_bet