from abc import ABC, abstractmethod
import pandas as pd

class Constraint(ABC):
    @abstractmethod
    def is_satisfied(self, df: pd.DataFrame) -> pd.Series: ...

