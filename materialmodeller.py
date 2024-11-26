from abc import ABC, abstractmethod
import numpy as np


def parabel_rektangel(
    e_c: float, e_c2: float, n: float, e_cu2: float, f_cd: float
) -> float:
    """Brukes til bøyning"""
    # Iht Figur 3.3 og Ligning (3.17)
    if e_c < e_cu2:
        return 0
    elif e_c <= e_c2:
        return f_cd
    elif e_c >= 0:
        return 0.0
    else:
        sigma_c = f_cd * (1 - (1 - e_c / e_c2) ** n)
        return sigma_c


def sargin_mat_model(
    e_c: float, e_c1: float, e_cu1: float, e_cm: float, f_cm: float
) -> float:
    """Brukes til énaksiell situasjon (søyle)"""
    # Iht Figur 3.2 og Ligning (3.14)
    k = 1.05 * e_cm * abs(e_c1) / f_cm
    eta = abs(e_c / e_c1)

    if e_c < e_cu1 or e_c > 0:
        return 0
    else:
        return f_cm * (k * eta - eta**2) / (1 + (k - 2) * eta) * np.sign(e_c)


class Material(ABC):
    """Generelt materialklasse"""

    @property
    def e_mod(self):
        return self.e_mod

    @e_mod.setter
    def e_mod(self, value):
        if value < 0:
            raise ValueError("Emod cannot be negative")
        self._e_mod = value

    @abstractmethod
    def get_stress(self, strain: float) -> float:
        """Gir spenning for en gitt tøyning"""
        pass

    @abstractmethod
    def get_f_yd(self) -> float:
        """Flytspenning"""

    @abstractmethod
    def get_eps_s_y(self) -> float:
        """flyttøyning"""


class CarbonMaterial(Material):
    """Karbonfiber, generell"""

    @abstractmethod
    def get_eps_s_y(self) -> float:
        """Flyttøyning"""
        pass

    @abstractmethod
    def get_e_modulus(self) -> float:
        """Elastisitetsmodul i N/mm2"""
        pass


class CarbonFiberDummy(CarbonMaterial):
    """Testmateriale for karbon"""

    def __init__(self) -> None:
        self.f_yk = 1e4
        self.gamma = 1.0
        self.e_s = 1 * 1e6
        self.f_yd = self.f_yk / self.gamma
        self.eps_s_y = self.f_yd / self.e_s

    def get_stress(self, strain: float) -> float:
        return strain * self.e_s

    def get_f_yd(self) -> float:
        """ " Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s_rebar(self) -> float:
        """Elastisitetsmodul"""
        return self.e_mod


class RebarMaterial(Material):
    """Abstrakt/generell armering"""

    @abstractmethod
    def get_eps_s_y(self) -> float:
        pass

    @abstractmethod
    def get_e_s_rebar(self) -> float:
        pass

    @abstractmethod
    def get_f_yd(self):
        pass


class RebarB500NC(RebarMaterial):
    """Vanlig armering"""

    def __init__(self) -> None:
        self.f_yk = 500
        self.gamma = 1.15
        self.e_s = 2 * 1e5
        self.f_yd = self.f_yk / self.gamma
        self.eps_s_y = self.f_yd / self.e_s

    def get_stress(self, strain: float) -> float:
        if strain < 0:
            return min(strain * self.e_s, self.f_yd)
        return max(strain * self.e_s, self.f_yd)

    def get_f_yd(self) -> float:
        """Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s_rebar(self) -> float:
        return self.e_s


class Tendon(RebarMaterial):
    """Spennarmering Y1860S7, flytspenning 1860, 7 strands"""

    def __init__(self) -> None:
        self.f_yk = 1860  # N/mm2 = MPa
        self.gamma = 1.15
        self.e_s = 2 * 1e5
        self.f_yd = self.f_yk / self.gamma
        self.eps_s_y = self.f_yd / self.e_s
        self.dia: float = 15.2  # mm
        self.area: float = 139  # mm2
        self.f_p: float = 0  # oppspenning i kN
        self.eps_0: float = 0  # initiell tøyning
        self.eps_s_y: float = 0.008087
        self.eps_s_u: float = 0.04

    def prestressd_to(self, force: float):
        """Sett oppspenningskraften i kN for instansen. Vanligvis 115 kN ish"""
        self.f_p = force
        sigma: float = force / self.area
        self.eps_0 = sigma / self.e_s

    def get_stress(self, strain: float) -> float:
        """Hent ut spenning fra tøyning. Enhet: N/mm2"""
        abs_strain: float = abs(strain)
        if abs_strain > self.eps_s_u:
            return 0
        if abs_strain > self.eps_s_y:
            return self.f_yd * np.sign(strain)
        return self.e_s * strain

    def get_area(self, vector: np.ndarray) -> np.ndarray:
        """Gir tilbake areal basert på antall pr lag"""
        return np.full(vector.shape, self.area)

    def get_f_yd(self) -> float:
        """Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s_rebar(self) -> float:
        return self.e_s


class ConcreteMaterial(Material):
    """Betongmateriale etter NS-EN 1992"""

    def __init__(self, f_ck: int, material_model: str = "Sargin") -> None:
        self.f_ck = -f_ck
        self.f_cd = self.f_ck / 1.5 * 0.85
        self.f_ctm: float = 0
        self.e_cm: float = 0
        self.p_r: float = 0.2
        self.f_cm: float = 0
        self.e_cy: float = 0
        self.e_cu: float = 0
        self.material_model = material_model
        self.f_ctm, self.e_cm = self.fetch_material_parameters()
        self.get_eps_cu_c_n()
        self.gamma = 1.5

    def get_stress(self, strain: float) -> float:
        if self.material_model == "Sargin":
            return sargin_mat_model(strain, self.e_cy, self.e_cu, self.e_cm, self.f_cm)
        return parabel_rektangel(strain, self.e_cy, self.n, self.e_cu, self.f_cd)

    def get_e_cu(self) -> float:
        """Bruddtøyning (minus i trykk)"""
        return self.e_cu

    def get_eps_cu_c_n(self):
        """Finn bruddtøyning og e_cx"""
        if self.material_model == "Parabola":
            (self.e_cu, self.e_cy, self.n) = self.get_eps_cu2_c2_n()
        elif self.material_model == "Bilinear":
            (self.e_cu, self.e_cy, self.n) = self.get_eps_cu3_c3()
        elif self.material_model == "Sargin":
            self.get_eps_cu1_c1()
        else:
            print("Error in material model name")

    def get_eps_cu1_c1(self):
        """Returns eps_c1 and eps_cu1 values for concrete type defined in Table 3.1 of NS-EN 1992-1-1"""
        self.f_cm = self.f_ck + 8.0
        self.e_cy = min(0.7 * self.f_cm**0.31, 2.8) / 1000.0
        if self.f_ck < 55.0:
            self.e_cu = -0.00350
        else:
            self.e_cu = -(2.8 + 27.0 * ((98.0 - self.f_cm) / 100.0) ** 4) / 1000.0

    def get_eps_cu2_c2_n(self):
        """Returns eps_c2 and eps_cu2 values for concrete type defined in Table 3.1 of NS-EN 1992-1-1"""
        if self.f_ck < 55.0:
            return (-0.00350, -0.00200, 2.0)
        if self.f_ck == 55.0:
            return (-0.00313, -0.00220, 1.75)
        if self.f_ck == 60.0:
            return (-0.00288, -0.00229, 1.59)
        if self.f_ck == 70.0:
            return (-0.00266, -0.00242, 1.44)
        if self.f_ck == 80.0:
            return (-0.00260, -0.00252, 1.4)
        return (-0.00260, -0.00260, 1.4)

    def get_eps_cu3_c3(self):
        """Returns eps_c3 and eps_cu3 values for concrete type defined in Table 3.1 of NS-EN 1992-1-1"""
        if self.f_ck < 55.0:
            return (-0.00350, -0.00175, 0.0)
        elif self.f_ck == 55.0:
            return (-0.00313, -0.00182, 0.0)
        elif self.f_ck == 60.0:
            return (-0.00288, -0.00188, 0.0)
        elif self.f_ck == 70.0:
            return (-0.00266, -0.00202, 0.0)
        elif self.f_ck == 80.0:
            return (-0.00260, -0.00216, 0.0)
        else:
            return (-0.00260, -0.00230, 0.0)

    def fetch_material_parameters(self):
        """Retrieves f_ctm and E_modbased on material strength"""
        properties = {
            20: (2.2, 3.0e4),
            25: (2.6, 3.1e4),
            30: (2.9, 3.3e4),
            35: (3.2, 3.4e4),
            40: (3.5, 3.5e4),
            45: (3.8, 3.6e4),
            50: (4.1, 3.7e4),
            55: (4.2, 3.8e4),
            60: (4.4, 3.9e4),
            70: (4.6, 4.1e4),
            80: (4.8, 4.2e4),
            90: (5.0, 4.4e4),
        }

        return properties.get(int(abs(self.f_ck)), (0.0, 0.0))

    def get_f_yd(self):
        return 0

    def get_eps_s_y(self) -> float:
        return 0


if __name__ == "__main__":
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    print("Hei")
    f_cd_temp = betong_b35.get_stress(0.0005)
    print("f_cd_temp:", f_cd_temp)
    # sum_f, sum_m, d_bet = integrate_cross_section(0.0035, 0, 0, 200, betong_b35, 300)
    # print(f"Force is {sum_f / 1000:.1f} kN")
