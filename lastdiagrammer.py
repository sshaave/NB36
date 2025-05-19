"""Import"""
from typing import Tuple
import numpy as np

class LastRespons():
    """Klasse som lager lastreponsdiagram for skjær og moment"""

    def __init__(self, length: float, q_uls: float, q_sls: float, q_init) -> None:
        """
        Lengtm  [m]         bjelkens lengde
        q_uls   [kN/m]      linjelast i ULS
        q_sls   [kN/m]      linjelast i SLS
        q_init  [kN/m]      linjelast når fiberarmering blir montert
        """
        self.length = length

        # Antall punkt hvor last regnes ut. Maks 200mm mellom hvert punkt, minimum 20 pkt langs bjelken
        self.antall_punkt = round(max(length / 0.2, 20)) + 1

        # Linjelaster
        self.q_uls = q_uls
        self.q_sls = q_sls
        self.q_init = q_init

        # Skjærdiagrammer
        self.v_uls = []
        self.v_sls = []
        self.v_init = []

        # Momentdiagrammer
        self.m_uls = []
        self.m_sls = []
        self.m_init = []

        self.calculate_response()

    def calculate_response(self) -> None:
        """Regner ut skjær og momentdiagrammer"""
        x_values: np.ndarray = np.linspace(0, self.length, self.antall_punkt)
        self.v_uls, self.m_uls = create_v_m_diagram(self.length, self.q_uls, x_values)
        self.v_sls, self.m_sls = create_v_m_diagram(self.length, self.q_sls, x_values)
        self.v_init, self.m_init = create_v_m_diagram(self.length, self.q_init, x_values)
    
    def get_m_uls(self) -> list[float]:
        return self.m_uls


def create_v_m_diagram(length: float, q: float, x: np.ndarray) -> Tuple[list[float], list[float]]:
    """Antar jevnt fordelt last"""

    # Skjær langs bjelken V(x) = qL/2 - qx
    v_values = q * length / 2 - q * x

    # Moment langs bjelken M(x) = qL/2 * x - q/2 * x^2
    m_values = x * q * length / 2 - q / 2 * x**2

    return v_values.tolist(), m_values.tolist()

if __name__ == "__main__":
    last_response: LastRespons = LastRespons(4.0, 12.0, 8.0, 4.0)
    m_uls = last_response.m_uls
    print([f"{m:.1f}" for m in m_uls])
