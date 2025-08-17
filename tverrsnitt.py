from numpy import ndarray

class Tverrsnitt():
    """Klasse for å representere et tverrsnitt med tilhørende egenskaper."""
    def __init__(
        self,
        height: float | ndarray,
        as_area_bot: float = 0,
        as_area_top: float = 0,
        d_bot: float = 0,
        d_top: float = 0,
        a_pre_bot: float = 0,
        a_pre_top: float = 0,
        d_pre_bot: float = 0,
        d_pre_top: float = 0,
        a_carbon: float = 0,
        d_carbon: float = 0,
    ) -> None:
        self.height = height
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
    def get_height(self) -> float:
        return self.height

    def get_height_max(self) -> float:
        """Returnerer maksimal høyde for tverrsnittet."""
        if isinstance(self.height, ndarray):
            return self.height.max()
        return self.height
    
    def get_as_area_bot(self) -> float:
        return self.as_area_bot

    def get_as_area_top(self) -> float:
        return self.as_area_top

    def get_d_bot(self) -> float:
        return self.d_bot

    def get_d_top(self) -> float:
        return self.d_top

    def get_a_pre_bot(self) -> float:
        return self.a_pre_bot

    def get_a_pre_top(self) -> float:
        return self.a_pre_top

    def get_d_pre_bot(self) -> float:
        return self.d_pre_bot

    def get_d_pre_top(self) -> float:
        return self.d_pre_top

    def get_a_carbon(self) -> float:
        return self.a_carbon

    def get_d_carbon(self) -> float:
        return self.d_carbon
  