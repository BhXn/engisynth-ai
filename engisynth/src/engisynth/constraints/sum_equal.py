import pandas as pd
from .base import Constraint

class SumEqual(Constraint):
    def __init__(self, cols, value=1.0, tol=1e-4):
        self.cols = cols
        self.value = value
        self.tol = tol

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.cols].sum(axis=1) - self.value).abs() < self.tol

