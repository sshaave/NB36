from abc import ABC, abstractmethod
import numpy as np


def parabel_rektangel(
    eps_c: float, eps_c2: float, n: float, eps_cu2: float, f_cd: float
) -> float:
    """Brukes til bøyning"""
    # Iht Figur 3.3 og Ligning (3.17)
    if eps_c < eps_cu2:
        return 0
    elif eps_c <= eps_c2:
        return f_cd
    elif eps_c >= 0:
        return 0.0
    else:
        sigma_c = f_cd * (1 - (1 - eps_c / eps_c2) ** n)
        return sigma_c


def sargin_mat_model(
    eps_c: float, eps_c1: float, eps_cu1: float, e_cm: float, f_cm: float
) -> float:
    """Brukes til énaksiell situasjon (søyle)"""
    # Iht Figur 3.2 og Ligning (3.14)
    k = 1.05 * e_cm * abs(eps_c1) / f_cm
    eta = abs(eps_c / eps_c1)

    if eps_c < eps_cu1 or eps_c > 0:
        return 0
    return f_cm * (k * eta - eta**2) / (1 + (k - 2) * eta) * np.sign(eps_c)


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
    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        """Gir spenning for en gitt tøyning"""

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

    @abstractmethod
    def get_e_s(self) -> float:
        """Elastisitetsmodul i N/mm2"""
        
    @abstractmethod
    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        """Gir spenning for en gitt tøyning"""

    @abstractmethod
    def set_eps_s_0_state(self, eps_s_0_state: float) -> None:
        """Sett tøyning ved montasje"""



class CarbonFiber(CarbonMaterial):
    """Testmateriale for karbon. Sika CarboDur M.
    https://nor.sika.com/no/losninger-innen-bygg/bygge/forsterkning-forankring/karbonfiberforsterkning/sika-carbodur-m.html"""

    def __init__(self) -> None:
        self.f_yk = 2900  # N/mm2
        self.gamma = 1.25
        self.e_s = 200000  # N/mm2
        self.f_yd = self.f_yk / self.gamma
        self.eps_s_y = 1.45 / 100  # Lineært elastiske til brudd, ingen plastisitet
        self.eps_s_u = self.eps_s_y
        self.eps_s_0_state = 0.0
    
    def get_eps_s_u(self) -> float:
        """Maks tillatte geometriske tøyning"""
        return self.eps_s_u - self.eps_s_0_state

    def get_eps_s_0_state(self) -> float:
        """Tøyning ved montering av fiberarmering"""
        return self.eps_s_0_state
    
    def set_eps_s_0_state(self, eps_s_0_state: float) -> None:
        """Setter tøyningsverdi ved montering av fiberarmering"""
        self.eps_s_0_state = eps_s_0_state
    
    def reset_0_state(self) -> None:
        """Setter verdien til 0"""
        self.eps_s_0_state = 0.
        
    def get_f_yd(self) -> float:
        """ " Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s(self) -> float:
        """Elastisitetsmodul"""
        return self.e_mod

    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        """Må ta hensyn til at karbonfiberen monteres når tverrsnittet potensielt er belastet.
        Tøyningen i karbonfiberen kan være lavere enn den geometriske tøyningen for tverrsnittet"""
        strain_carbon_fiber = strain - self.eps_s_0_state
        
        return 0 if abs(strain_carbon_fiber) > self.eps_s_u else strain_carbon_fiber * self.e_s


class RebarMaterial(Material):
    """Abstrakt/generell armering"""

    @abstractmethod
    def get_eps_s_y(self) -> float:
        pass
    
    @abstractmethod
    def get_eps_s_u(self) -> float:
        pass
        
    @abstractmethod
    def get_e_s_rebar(self) -> float:
        """E-modulus i N/mm2"""
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
        self.eps_s_u = 2 / 100

    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        f_yk_yd = self.f_yk if is_ck_not_cd else self.f_yd
            
        if strain < 0:
            return max(strain * self.e_s, -f_yk_yd)
        return min(strain * self.e_s, f_yk_yd)

    def get_f_yd(self) -> float:
        """Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s_rebar(self) -> float:
        return self.e_s

    def get_eps_s_u(self) -> float:
        return self.eps_s_u


class RebarB400NC(RebarMaterial):
    """Vanlig armering"""

    def __init__(self) -> None:
        self.f_yk = 400
        self.gamma = 1.15
        self.e_s = 2 * 1e5
        self.f_yd = self.f_yk / self.gamma
        self.eps_s_y = self.f_yd / self.e_s
        self.eps_s_u = 2 / 100

    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        f_yk_yd = self.f_yk if is_ck_not_cd else self.f_yd
        
        if strain < 0:
            return max(strain * self.e_s, -f_yk_yd)
        return min(strain * self.e_s, f_yk_yd)

    def get_f_yd(self) -> float:
        """Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_e_s_rebar(self) -> float:
        return self.e_s

    def get_eps_s_u(self) -> float:
        return self.eps_s_u


class Tendon(RebarMaterial):
    """Spennarmering Y1860S7, flytspenning 1860, 7 strands"""

    def __init__(self) -> None:
        self.f_yk = 1700  # N/mm2 = MPa
        self.gamma = 1.15
        self.e_s = 195000
        self.f_yd = self.f_yk / self.gamma
        self.dia: float = 11.3  # mm
        self.area: float = 100  # mm2
        self.f_p: float = 0  # oppspenning i kN
        self.eps_0: float = 0  # initiell tøyning
        self.eps_s_y: float = self.f_yk / self.e_s / self.gamma
        self.eps_s_u: float = 0.037 * 0.9

    def set_fp(self, force: float):
        """Sett oppspenningskraften i kN for instansen. Vanligvis 115 kN ish"""
        self.f_p = force * 1000  # self.f_p i N
        sigma_0: float = self.f_p / self.area
        self.eps_0 = sigma_0 / self.e_s
    
    def get_fp(self) -> float:
        """Hent ut forspenningskraft i kN"""
        return self.f_p / 1000

    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        """Hent ut spenning fra tøyning. Enhet: N/mm2"""
        f_yk_yd = self.f_yk if is_ck_not_cd else self.f_yd
        total_strain: float = strain + self.eps_0
        if abs(total_strain) > self.eps_s_u:
            return 0
    
        if total_strain < 0:
                return max(total_strain * self.e_s, -f_yk_yd)
        return min(total_strain * self.e_s, f_yk_yd)


    def get_area(self, vector: np.ndarray) -> np.ndarray:
        """Gir tilbake areal basert på antall pr lag"""
        return vector * self.area

    def get_antall_vec(self, vector: np.ndarray) -> np.ndarray:
        """Gir tilbake antall i hvert lag basert på areal"""
        return np.floor(vector / self.area)

    def get_f_yd(self) -> float:
        """Dimensjonerende flytespenning"""
        return self.f_yd

    def get_eps_s_y(self) -> float:
        return self.eps_s_y

    def get_eps_s_u(self) -> float:
        return self.eps_s_u
    
    def get_e_s_rebar(self) -> float:
        return self.e_s

    def get_prestress(self) -> float:
        """Effektiv forspenningskraft i N"""
        return self.f_p * 1e3

    def get_eps_f_p(self) -> float:
        """Få tøyning pga forspenning"""
        return self.eps_0

    def get_max_external_strain(self) -> float:
        """Regn ut maks tilleggstøyning etter forspenning er satt på"""
        return self.eps_s_u - self.eps_0


class ConcreteMaterial(Material):
    """Betongmateriale etter NS-EN 1992"""

    def __init__(self, f_ck: int, material_model: str = "Sargin") -> None:
        self.f_ck = -f_ck
        self.f_cd = self.f_ck / 1.5 * 0.85
        self.f_ctm: float = 0
        self.e_cm: float = 0
        self.p_r: float = 0.2
        self.f_cm: float = 0
        self.eps_cy: float = 0
        self.eps_cu: float = 0
        self.material_model = material_model
        self.f_ctm, self.e_cm = self.fetch_material_parameters()
        self.get_eps_cu_c_n()
        self.gamma = 1.5
        self.creep_eff: float = 0.0  # Effektivt kryptall

    def get_stress(self, strain: float, is_ck_not_cd: bool = True) -> float:
        # Sargin er ikke klar for ULS, men det gjør ingenting for bjelker
        eps_cy = self.eps_cy * (1 + self.creep_eff)
        eps_cu = self.eps_cu * (1 + self.creep_eff)
        if self.material_model == "Sargin":
            return sargin_mat_model(strain, eps_cy, eps_cu, self.e_cm, self.f_cm)
        f_ck_cd = self.f_ck if is_ck_not_cd else self.f_cd
        return parabel_rektangel(strain, eps_cy, self.n, eps_cu, f_ck_cd)

    def change_material_model(self, material_model: str):
        """Endrer materialmodell for betong"""
        self.material_model = material_model
        self.get_eps_cu_c_n()

    def get_eps_cu(self) -> float:
        """Bruddtøyning (minus i trykk)"""
        return self.eps_cu

    def get_eps_cu_c_n(self):
        """Finn bruddtøyning og eps_cx"""
        if self.material_model == "Parabola":
            (self.eps_cu, self.eps_cy, self.n) = self.get_eps_cu2_c2_n()
        elif self.material_model == "Bilinear":
            (self.eps_cu, self.eps_cy, self.n) = self.get_eps_cu3_c3()
        elif self.material_model == "Sargin":
            self.get_eps_cu1_c1()
        else:
            print("Error in material model name")

    def get_eps_cu1_c1(self):
        """Returns eps_c1 and eps_cu1 values for concrete type defined in Table 3.1 of NS-EN 1992-1-1"""
        self.f_cm = self.f_ck - 8.0
        print("f_cm:", self.f_cm)
        self.eps_cy = min(0.7 * (-self.f_cm)**0.31, 2.8) / 1000.0
        if self.f_ck < 55.0:
            self.eps_cu = -0.00350
        else:
            self.eps_cu = -(2.8 + 27.0 * ((98.0 - self.f_cm) / 100.0) ** 4) / 1000.0

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
    
    def set_creep(self, creep_eff: float) -> None:
        """Setter effektivt kryptall"""
        if creep_eff < 0:
            raise ValueError("Creep_eff cannot be negative")
        self.creep_eff = creep_eff

    def get_creep(self) -> float:
        """Hent effektivt kryptall"""
        return self.creep_eff
    
    def get_eps_cu_eff(self) -> float:
        """Hent effektiv bruddtøyning"""
        return self.eps_cu * (1 + self.creep_eff)


if __name__ == "__main__":
    betong_b35: ConcreteMaterial = ConcreteMaterial(35, material_model="Parabola")
    print("Hei")
    f_cd_temp = betong_b35.get_stress(0.0005)
    print("f_cd_temp:", f_cd_temp)
    # sum_f, sum_m, d_bet = integrate_cross_section(0.0035, 0, 0, 200, betong_b35, 300)
    # print(f"Force is {sum_f / 1000:.1f} kN")
