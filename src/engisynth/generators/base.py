from abc import ABC, abstractmethod
import pandas as pd

class BaseGenerator(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs): ...
    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...

