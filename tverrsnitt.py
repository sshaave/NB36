import numpy as np
from numpy import ndarray

class Tverrsnitt():
    """Klasse for å representere et tverrsnitt med tilhørende egenskaper."""
    def __init__(
        self,
        height: float | ndarray,
        width,
        as_area_bot: ndarray = np.array([]),
        as_area_top: ndarray = np.array([]),
        d_bot: ndarray = np.array([]),
        d_top: ndarray = np.array([]),
        a_pre_bot: ndarray = np.array([]),
        a_pre_top: ndarray = np.array([]),
        d_pre_bot: ndarray = np.array([]),
        d_pre_top: ndarray = np.array([]),
        a_carbon: ndarray = np.array([]),
        d_carbon: ndarray = np.array([])
    ) -> None:
        self.height = height
        self.height_i = height if isinstance(height, float) else height[0]
        self.width = width
        self.as_area_bot = as_area_bot
        self.as_area_top = as_area_top
        self.d_bot = d_bot
        self.d_top = d_top
        self.a_pre_bot = a_pre_bot
        self.a_pre_top = a_pre_top
        self.d_pre_bot = d_pre_bot
        self.d_pre_top = d_pre_top
        self.a_carbon = a_carbon
        self.d_carbon = d_carbon

    def get_height(self) -> float | ndarray:
        """Gir tilbake høyde (kan være vektor)"""
        return self.height

    def get_height_max(self) -> float:
        """Returnerer maksimal høyde for tverrsnittet."""
        if isinstance(self.height, ndarray):
            return self.height.max()
        return self.height

    def set_height_i(self, i: int) -> None:
        """Setter hvilken høyde get_height_i henter verdi fra"""
        if isinstance(self.height, ndarray):
            if i >= len(self.height):
                raise IndexError("Index i er utenfor rekkevidde for height-vektoren. Sjekk minstekrav i hjelpemetoder.py linje 236.")
            self.height_i = self.height[i]
        else:
            self.height_i = self.height

    def set_height_to_max(self) -> None:
        """Setter høyden til maksimal verdi"""
        if isinstance(self.height, ndarray):
            self.height_i = self.height.max()
        else:
            self.height_i = self.height

    def get_height_i(self) -> float:
        """ Gir høyden for satt snitt"""
        return self.height_i

    def get_width(self, height: float) -> float:
        """ Gir tilbake bredde for en høyde, brukerdefinert funksjon"""
        if callable(self.width):
            return self.width(height)
        return self.width

    def get_as_area_bot(self) -> ndarray:
        return self.as_area_bot

    def get_as_area_top(self) -> ndarray:
        return self.as_area_top

    def get_d_bot(self) -> ndarray:
        return self.d_bot

    def get_d_top(self) -> ndarray:
        return self.d_top

    def get_d0_bot(self) -> float:
        """ Gir tyngdepunktet til ytterste strekkmateriale"""
        if len(self.d_pre_bot) == 0:
            d_pre_bot = 0
        else:
            d_pre_bot = self.d_pre_bot[0]

        if len(self.d_bot) == 0:
            d_bot_slakkarmering = 0
        else:
            d_bot_slakkarmering = self.d_bot[0]

        if len(self.d_carbon) == 0:
            d_carbon = 0
        else:
            d_carbon = self.d_carbon[0]
        return max(d_pre_bot, d_bot_slakkarmering, d_carbon)

    def get_d_top_avg(self) -> float:
        """UTEN KF. Returnerer tyngdepunktet av d for armeringen i OK (ikke bare ytterste laget)."""
        d_area = 0.0
        sum_area = 0.0
        for d, a_i in zip(self.d_top, self.as_area_top):
            d_area += d * a_i
            sum_area += a_i
        for d, a_i in zip(self.d_pre_top, self.a_pre_top):
            d_area += d * a_i
            sum_area += a_i
        if sum_area == 0:
            return 0.0
        d_avg_top = d_area / sum_area
        return d_avg_top

    def get_d_bot_avg(self) -> float:
        """UTEN KF. Returnerer tyngdepunktet av d for armeringen i UK (ikke bare ytterste laget)."""
        d_area = 0.0
        sum_area = 0.0
        for d, a_i in zip(self.d_bot, self.as_area_bot):
            d_area += d * a_i
            sum_area += a_i
        for d, a_i in zip(self.d_pre_bot, self.a_pre_bot):
            d_area += d * a_i
            sum_area += a_i
        if sum_area == 0:
            return 0.0
        d_avg_bot = d_area / sum_area
        return d_avg_bot

    def get_d_bot_avg_cf(self) -> float:
        """Returnerer tyngdepunktet av d for armeringen i UK (ikke bare ytterste laget)."""
        d_area = 0.0
        sum_area = 0.0
        for d, a_i in zip(self.d_bot, self.as_area_bot):
            d_area += d * a_i
            sum_area += a_i
        for d, a_i in zip(self.d_pre_bot, self.a_pre_bot):
            d_area += d * a_i
            sum_area += a_i
        for d, a_i in zip(self.d_carbon, self.a_carbon):
            d_area += d * a_i
            sum_area += a_i
        if sum_area == 0:
            return 0.0
        d_avg_bot = d_area / sum_area
        return d_avg_bot

    def get_a_bot_sum(self) -> float:
        return np.sum(self.as_area_bot) + np.sum(self.a_pre_bot)

    def get_a_top_sum(self) -> float:
        return np.sum(self.as_area_top) + np.sum(self.a_pre_top)

    def get_a_pre_bot(self) -> ndarray:
        return self.a_pre_bot

    def get_a_pre_top(self) -> ndarray:
        return self.a_pre_top

    def get_d_pre_bot(self) -> ndarray:
        return self.d_pre_bot

    def get_d_pre_top(self) -> ndarray:
        return self.d_pre_top

    def get_a_carbon(self) -> ndarray:
        return self.a_carbon

    def get_d_carbon(self) -> ndarray:
        return self.d_carbon

    def __str__(self):
        return (
            f"Tverrsnitt:\n"
            f"  Height: {self.height}\n"
            f"  As area bot: {self.as_area_bot}\n"
            f"  As area top: {self.as_area_top}\n"
            f"  d bot: {self.d_bot}\n"
            f"  d top: {self.d_top}\n"
            f"  a pre bot: {self.a_pre_bot}\n"
            f"  a pre top: {self.a_pre_top}\n"
            f"  d pre bot: {self.d_pre_bot}\n"
            f"  d pre top: {self.d_pre_top}\n"
            f"  a carbon: {self.a_carbon}\n"
            f"  d carbon: {self.d_carbon}\n"
            f"  Sum armering i OK: {self.get_a_top_sum():.1f}\n"
            f"  Sum armering i uK: {self.get_a_bot_sum():.1f}\n"
        )
