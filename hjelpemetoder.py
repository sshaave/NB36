"""Hjelpemetoder"""
from typing import Tuple

def eps_ok_uk_to_c_and_s(
    eps_ok: float,
    eps_uk: float,
    height: float,
    strekk_uk: bool,
    strekk_ok: bool,
    d0_strekk: float,
    d0_trykk: float
) -> Tuple[float, float, float]:
    """Metode for å gjøre om over- og underkantøyningter. Returerer betongtøyning, armeringstøyning og alpha."""
    tolerance: float = 1e-8
    if abs(eps_ok - eps_uk) < tolerance:
        return eps_ok, eps_ok, 0.0

    d_eps_dx: float = (eps_uk - eps_ok) / height

    if strekk_uk and not strekk_ok:
        # Strekk UK og trykk OK
        eps_s = eps_uk - d_eps_dx * (height - d0_strekk)
        alpha = -eps_ok / (eps_s - eps_ok)
        return eps_ok, eps_s, alpha

    elif not strekk_uk and strekk_ok:
        # Strekk OK og trykk UK
        eps_s = eps_uk - d_eps_dx * (height - d0_strekk)
        alpha = -eps_uk / (eps_s - eps_uk)
        return eps_uk, eps_s, alpha

    elif strekk_uk and strekk_ok:
        # Strekk i hele tverrsnittet
        eps_s_uk = eps_uk - d_eps_dx * (height - d0_strekk)
        eps_s_ok = eps_ok - d_eps_dx * (height - d0_trykk)
        return eps_s_ok, eps_s_uk, 0.0

    else:
        # Trykk i hele tverrsnittet
        if eps_ok < eps_uk:
            # Mest trykk i OK
            alpha_d = -eps_ok / d_eps_dx
        else:
            # Mest trykk i UK
            alpha_d = -eps_uk / d_eps_dx
        alpha = alpha_d / d0_strekk
        return min(eps_ok, eps_uk), 0.0, alpha
    